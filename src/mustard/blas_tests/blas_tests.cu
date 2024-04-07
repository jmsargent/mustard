#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>
#include <nvshmem.h>
#include <nvshmemx.h>
// #include <fmt/core.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <tuple>
#include <vector>

#include "../../include/argh.h"
// #include "../utilities/cudaUtilities.hpp"

#define FLAGS_SUBG_COUNT 0

size_t N = 15 * 1;
size_t B = N / 5;
size_t T = N / B;
int myPE;
int verbose = 0;
int workspace = 1;
int smLimit = 10;
int runs = 1;


template <typename T>
void __check(T result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        fprintf(stderr, "CUDA error at %s:%d code=%d \"%s\" \n", file, line, static_cast<unsigned int>(result), func);
        exit(EXIT_FAILURE);
    }
}

void checkAndSetP2Paccess(int numGPUs)
{
    for (int i = 0; i < numGPUs; i++)
    {
        cudaSetDevice(i);

        for (int j = 0; j < numGPUs; j++)
        {
            int access;
            if (i != j)
            {
                cudaDeviceCanAccessPeer(&access, i, j);
                if (!access)
                    printf("Device=%d CANNOT Access Peer Device=%d\n", i, j);
                // printf("Device=%d %s Access Peer Device=%d\n", i, access ? "CAN" : "CANNOT", j);
                if (access)
                    cudaDeviceEnablePeerAccess(j, 0);
            }
        }
    }
}

#define checkCudaErrors(val) __check((val), #val, __FILE__, __LINE__)

void showMemUsage()
{
    // show memory usage of GPU
    size_t free_byte;
    size_t total_byte;

    cudaError_t cuda_status = cudaMemGetInfo(&free_byte, &total_byte);

    if (cudaSuccess != cuda_status)
    {
        printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status));
        exit(1);
    }

    double free_db = (double)free_byte;
    double total_db = (double)total_byte;
    double used_db = total_db - free_db;
    printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
            used_db / 1024.0 / 1024.0, free_db / 1024.0 / 1024.0, total_db / 1024.0 / 1024.0);
}

__global__ void warmUp()
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;
    ib += ia + static_cast<float>(tid);
}

void warmUpCudaDevice()
{
    warmUp<<<32, 32>>>();
    cudaDeviceSynchronize();
}

void initializeCudaDevice(bool displayDeviceInfo)
{
    // checkCudaErrors(cudaSetDevice(0));

    if (displayDeviceInfo)
    {
        cudaDeviceProp deviceProp;
        checkCudaErrors(cudaGetDeviceProperties(&deviceProp, 0));
        printf("GPU Device %d: %s\n", 0, deviceProp.name);
        printf("Compute Capability: %d.%d\n\n", deviceProp.major, deviceProp.minor);
    }

    warmUpCudaDevice();
}

class CudaEventClock
{
public:
    CudaEventClock();
    ~CudaEventClock();
    void start(cudaStream_t stream = 0);
    void end(cudaStream_t stream = 0);
    float getTimeInSeconds();

private:
    cudaEvent_t startEvent, endEvent;
};

CudaEventClock::CudaEventClock()
{
    checkCudaErrors(cudaEventCreate(&this->startEvent));
    checkCudaErrors(cudaEventCreate(&this->endEvent));
}

CudaEventClock::~CudaEventClock()
{
    checkCudaErrors(cudaEventDestroy(this->startEvent));
    checkCudaErrors(cudaEventDestroy(this->endEvent));
}

void CudaEventClock::start(cudaStream_t stream)
{
    checkCudaErrors(cudaEventRecord(this->startEvent, stream));
}

void CudaEventClock::end(cudaStream_t stream)
{
    checkCudaErrors(cudaEventRecord(this->endEvent, stream));
}

float CudaEventClock::getTimeInSeconds()
{
    float time;
    checkCudaErrors(cudaEventElapsedTime(&time, this->startEvent, this->endEvent));
    return time * 1e-3f;
}


// Credit to: https://math.stackexchange.com/questions/357980/how-to-generate-random-symmetric-positive-definite-matrices-using-matlab
void generateRandomSymmetricPositiveDefiniteMatrix(double *h_A, const size_t n)
{
    // srand(time(NULL));
    srand(420);

    double *h_A_temp = (double *)malloc(n * n * sizeof(double));

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            h_A_temp[i * n + j] = (float)rand() / (float)RAND_MAX;

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            h_A[i * n + j] = 0.5 * (h_A_temp[i * n + j] + h_A_temp[j * n + i]);

    for (int i = 0; i < n; i++)
        h_A[i * n + i] = h_A[i * n + i] + n;
}

void printSquareMatrix(double *h_A, const size_t n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (j != 0)
                std::cout << " ";
            std::cout << std::setw(6) << std::setprecision(3) << h_A[i * n + j];
        }
        std::cout << std::endl;
    }
}

// Set upper triangle entries (excluding diagonal entries) in column-major order to zero.
// Then, transpose to row-major order.
void cleanCusolverLUDecompositionResult(double *L, double *U, const int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = i; j < n; j++)
        {
            std::swap(L[i + j * n], L[i * n + j]);
            U[i * n + j] = L[i * n + j];
            L[i * n + j] = 0;
        }
        L[i * n + i] = 1;
    }
}
    
typedef std::pair<int, int> MatrixTile;

class TiledLUGraphCreator
{
public:
    cudaGraph_t *subgraphs;

    TiledLUGraphCreator(cudaStream_t stream, cudaGraph_t graph, bool subgraph = false, int totalNodes = 1) : stream(stream), graph(graph)
    {
        this->lastModifiedTile = std::make_pair(-1, -1);
        this->subgraph = subgraph;
        this->subgraphs = new cudaGraph_t[totalNodes];
        this->index_counter = 0;
    }

    void beginCaptureOperation(MatrixTile tileToWrite, std::initializer_list<MatrixTile> tilesToRead)
    {            
        auto tiles = std::vector<MatrixTile>(tilesToRead);
        tiles.push_back(tileToWrite);

        this->lastModifiedTile = tileToWrite;

        if (!this->subgraph) {
            auto dependencies = this->getDependencies(tiles);
            this->lastDependencies = dependencies;
            checkCudaErrors(cudaStreamBeginCaptureToGraph(this->stream, this->graph, dependencies.data(), 
                                                          nullptr, dependencies.size(), cudaStreamCaptureModeGlobal));
        } else {
            checkCudaErrors(cudaGraphCreate(&this->subgraphs[index_counter], 0)); 
            // std::cout << "Start capturing subgraph " << index_counter << std::endl;
            checkCudaErrors(cudaStreamBeginCaptureToGraph(this->stream, this->subgraphs[index_counter], nullptr, 
                                                          nullptr, 0, cudaStreamCaptureModeGlobal));
        }
    }

    void endCaptureOperation()
    {
        assert(this->lastModifiedTile.first != -1 && this->lastModifiedTile.second != -1);
        if (!this->subgraph) {
            checkCudaErrors(cudaStreamEndCapture(this->stream, &this->graph));
            this->tileLastModifiedByMap[this->lastModifiedTile] = this->getTailOfLastCapturedNodeChain();
        } else {
            checkCudaErrors(cudaStreamEndCapture(this->stream, &(this->subgraphs[index_counter])));
            this->tileIndexByMap[this->lastModifiedTile] = this->index_counter;
            // std::cout << "End capturing subgraph " << index_counter << std::endl;
            this->index_counter++;
        }
        this->lastModifiedTile = std::make_pair(-1, -1);
    };

private:
    std::map<MatrixTile, cudaGraphNode_t> tileLastModifiedByMap;
    std::map<MatrixTile, int> tileIndexByMap;
    std::map<cudaGraphNode_t, bool> visited;
    cudaStream_t stream;
    cudaGraph_t graph;
    MatrixTile lastModifiedTile;
    std::vector<cudaGraphNode_t> lastDependencies;
    int index_counter;
    bool subgraph;

    // auto getDependencies(std::vector<MatrixTile> tiles, bool) {
    // }

    std::vector<cudaGraphNode_t> getDependencies(std::vector<MatrixTile> tiles)
    {
        std::vector<cudaGraphNode_t> dependencies;
        for (auto tile : tiles)
        {
            auto it = this->tileLastModifiedByMap.find(tile);
            if (it != this->tileLastModifiedByMap.end())
            {
                dependencies.push_back(it->second);
            }
        }

        auto dedupedEnd = std::unique(dependencies.begin(), dependencies.end());
        dependencies.resize(std::distance(dependencies.begin(), dedupedEnd));

        return dependencies;
    }
    cudaGraphNode_t getTail(cudaGraph_t graph){
        size_t numEdges;
        checkCudaErrors(cudaGraphGetEdges(graph, nullptr, nullptr, &numEdges));
        if (numEdges == 0) 
        {
            size_t numNodes = 1;
            auto nodes = std::make_unique<cudaGraphNode_t[]>(1);
            checkCudaErrors(cudaGraphGetNodes(graph, nodes.get(), &numNodes));
            return nodes[0];
        }
        auto from = std::make_unique<cudaGraphNode_t[]>(numEdges);
        auto to = std::make_unique<cudaGraphNode_t[]>(numEdges);
        checkCudaErrors(cudaGraphGetEdges(graph, from.get(), to.get(), &numEdges));

        std::map<cudaGraphNode_t, bool> hasOutGoingEdge;
        std::set<cudaGraphNode_t> noOutGoingEdgeNodes;
        for (int i = 0; i < numEdges; i++)
        {
            hasOutGoingEdge[from[i]] = true;
            noOutGoingEdgeNodes.erase(from[i]);
            if (!hasOutGoingEdge[to[i]])
                noOutGoingEdgeNodes.insert(to[i]);
        }

        assert(noOutGoingEdgeNodes.size() == 1);

        return *noOutGoingEdgeNodes.begin();
    }

    cudaGraphNode_t getTailOfLastCapturedNodeChain()
    {
        if (lastDependencies.size() == 0)
        {
            return getTail(this->graph);
        }
        else
        {
            auto nodeBeforeChain = lastDependencies[0];
            size_t numDependentNodes;
            checkCudaErrors(cudaGraphNodeGetDependentNodes(nodeBeforeChain, nullptr, &numDependentNodes));

            assert(numDependentNodes > 0);

            auto dependentNodes = std::make_unique<cudaGraphNode_t[]>(numDependentNodes);
            checkCudaErrors(cudaGraphNodeGetDependentNodes(nodeBeforeChain, dependentNodes.get(), &numDependentNodes));

            cudaGraphNode_t chainBeginningNode;
            for (int i = 0; i < numDependentNodes; i++)
            {
                if (!visited[dependentNodes[i]])
                {
                    chainBeginningNode = dependentNodes[i];
                    break;
                }
            }

            auto u = chainBeginningNode;
            while (true)
            {
                visited[u] = true;
                checkCudaErrors(cudaGraphNodeGetDependentNodes(u, nullptr, &numDependentNodes));
                if (numDependentNodes == 0)
                    break;

                assert(numDependentNodes == 1);

                cudaGraphNode_t v;
                checkCudaErrors(cudaGraphNodeGetDependentNodes(u, &v, &numDependentNodes));
                u = v;
            }

            return u;
        }
    }
};

double* getMatrixBlock(double* d_matrix, int i, int j)
{
    return d_matrix + i * B + j * B * N;
};

void copyMatrixBlock(double* to, double* from, int i, int j, int B){
    double* to_start = getMatrixBlock(to, i, j);
    double* from_start = getMatrixBlock(from, i, j);
    for (int row = 0; row < B; row++) {
        checkCudaErrors(cudaMemcpy(to_start + N*row, from_start + N*row, B * sizeof(double), cudaMemcpyDeviceToDevice));
    }
}

void tiledLU(bool remote, bool copy, bool subgraph, bool dot)
{
    // Initialize data
    auto originalMatrix = std::make_unique<double[]>(N * N); // Column-major
    generateRandomSymmetricPositiveDefiniteMatrix(originalMatrix.get(), N);

    // Copy to device
    double *d_matrix;
    double *d_matrices;
    double *d_matrix_remote;
    d_matrices = (double *)nvshmem_malloc(N * N * sizeof(double));
    d_matrix = (double *)nvshmem_ptr(d_matrices, (int)remote);
    d_matrix_remote = (double *)nvshmem_ptr(d_matrices, 1);
    if (myPE == (int)remote)
        checkCudaErrors(cudaMemcpy(d_matrix, originalMatrix.get(), N * N * sizeof(double), cudaMemcpyHostToDevice));
    if (copy) {
        d_matrix = (double *)nvshmem_ptr(d_matrices, 0);
    }


    // Initialize libraries
    cusolverDnHandle_t cusolverDnHandle;
    cusolverDnParams_t cusolverDnParams;
    cublasHandle_t cublasHandle;
    checkCudaErrors(cusolverDnCreate(&cusolverDnHandle));
    checkCudaErrors(cusolverDnCreateParams(&cusolverDnParams));
    checkCudaErrors(cublasCreate(&cublasHandle));
    // checkCudaErrors(cublasLoggerConfigure(verbose, verbose, 0, NULL));
    checkCudaErrors(cublasSetSmCountTarget(cublasHandle, smLimit));

    // Prepare constants
    double one = 1.0;
    double minusOne = -1.0;

    // Prepare buffer for potrf
    // size_t workspaceInBytesOnDevice, workspaceInBytesOnHost;
    int workspaceInBytesOnDevice;
        

    checkCudaErrors(cusolverDnDgetrf_bufferSize(
                    cusolverDnHandle,
                    B,
                    B,
                    d_matrix,
                    N,
                    &workspaceInBytesOnDevice));

    // void *h_workspace, *d_workspace_cusolver;
    double *d_workspace_cusolver;
    int workspaces = T;//(T-1)*(T-1);
    void **d_workspace_cublas = (void **)malloc(sizeof(void *)*workspaces);
    int *d_info;
    // checkCudaErrors(cudaMalloc(&h_workspace, workspaceInBytesOnHost));
    checkCudaErrors(cudaMalloc(&d_workspace_cusolver, workspaceInBytesOnDevice*1024));
    int cublasWorkspaceSize = 1024*workspace; // (B/256+1)*B*256*4;
    for (int i = 0; i < workspaces; i++) {
        checkCudaErrors(cudaMalloc(&d_workspace_cublas[i], cublasWorkspaceSize));
    }
    checkCudaErrors(cudaMalloc(&d_info, sizeof(int)));

    cudaGraph_t graph;
    checkCudaErrors(cudaGraphCreate(&graph, 0));

    int totalNodes = 4;

    if (verbose) {
        std::cout << "totalNodes=" << totalNodes << std::endl;
        std::cout << "bufferSize=" << workspaceInBytesOnDevice << std::endl;
        std::cout << "tileSize=" << cublasWorkspaceSize << std::endl;
    }

    cudaStream_t s;
    checkCudaErrors(cudaStreamCreate(&s));

    checkCudaErrors(cusolverDnSetStream(cusolverDnHandle, s));
    checkCudaErrors(cublasSetStream(cublasHandle, s));
    checkCudaErrors(cublasSetWorkspace(cublasHandle, d_workspace_cublas[0], cublasWorkspaceSize));

    auto tiledLUGraphCreator = std::make_unique<TiledLUGraphCreator>(s, graph, subgraph, totalNodes);

    for (int k = 0; k < 1; k++)
    {
        // A[k][k] = GETRF(A[k][k])
        // L[k][k]*U[k][k] = A[k][k]
        tiledLUGraphCreator->beginCaptureOperation(
            std::make_pair(k, k),
            {std::make_pair(k, k)});
        // if (copy) copyMatrixBlock(d_matrix, d_matrix_remote, k, k, B);
        if (copy) cudaMemcpy2DAsync(getMatrixBlock(d_matrix, k, k), 
                                    sizeof(double) * N,
                                    getMatrixBlock(d_matrix_remote, k, k), 
                                    sizeof(double) * N, 
                                    sizeof(double) * B, 
                                    B, cudaMemcpyDeviceToDevice, s);
        checkCudaErrors(cusolverDnDgetrf(
            cusolverDnHandle,
            B,
            B,
            getMatrixBlock(d_matrix, k, k),
            N,
            d_workspace_cusolver,
            NULL,
            d_info));
        if (copy) cudaMemcpy2DAsync(getMatrixBlock(d_matrix_remote, k, k), 
                                    sizeof(double) * N,
                                    getMatrixBlock(d_matrix, k, k), 
                                    sizeof(double) * N, 
                                    sizeof(double) * B, 
                                    B, cudaMemcpyDeviceToDevice, s);
        tiledLUGraphCreator->endCaptureOperation();

        for (int i = k + 1; i < 2; i++)
        {
            // L[i][k] = TRSM(A[i][k], A[k][k]) // the U part of A[k][k]
            // seems like only these need a separate workspace
            checkCudaErrors(cublasSetWorkspace(cublasHandle, d_workspace_cublas[i], cublasWorkspaceSize));
            tiledLUGraphCreator->beginCaptureOperation(
                std::make_pair(k, i),
                {std::make_pair(k, k), std::make_pair(k, i)});
            //if (copy) copyMatrixBlock(d_matrix, d_matrix_remote, k, i, B);
            //if (copy) copyMatrixBlock(d_matrix, d_matrix_remote, k, k, B);
            if (copy) cudaMemcpy2DAsync(getMatrixBlock(d_matrix, k, i), 
                                        sizeof(double) * N,
                                        getMatrixBlock(d_matrix_remote, k, i), 
                                        sizeof(double) * N, 
                                        sizeof(double) * B, 
                                        B, cudaMemcpyDeviceToDevice, s);
            checkCudaErrors(cublasDtrsm(
                cublasHandle,
                CUBLAS_SIDE_LEFT, // used to be right for cholesky
                CUBLAS_FILL_MODE_LOWER,
                CUBLAS_OP_N,// CUBLAS_OP_T for cholesky
                CUBLAS_DIAG_UNIT, // CUBLAS_DIAG_NON_UNIT for cholesky
                B, B,
                &one,
                getMatrixBlock(d_matrix, k, k), N, // k + k * N;
                getMatrixBlock(d_matrix, k, i), N)); // k + (i + B) * N;
            if (copy) cudaMemcpy2DAsync(getMatrixBlock(d_matrix_remote, k, i), 
                                        sizeof(double) * N,
                                        getMatrixBlock(d_matrix, k, i), 
                                        sizeof(double) * N, 
                                        sizeof(double) * B, 
                                        B, cudaMemcpyDeviceToDevice, s);
            tiledLUGraphCreator->endCaptureOperation();

        }
        checkCudaErrors(cublasSetWorkspace(cublasHandle, d_workspace_cublas[0], cublasWorkspaceSize));

        for (int i = k + 1; i < 2; i++)
        {
            // U[k][i] = TRSM(A[k][k], A[k][i]) // the L part of A[k][k]
            // checkCudaErrors(cublasSetWorkspace(cublasHandle, d_workspace_cublas[i-1 + T], workspaceInBytesOnDevice));
            tiledLUGraphCreator->beginCaptureOperation(
                std::make_pair(i, k),
                {std::make_pair(k, k), std::make_pair(i, k)});
            //if (copy) copyMatrixBlock(d_matrix, d_matrix_remote, i, k, B);
            //if (copy) copyMatrixBlock(d_matrix, d_matrix_remote, k, k, B);
            if (copy) cudaMemcpy2DAsync(getMatrixBlock(d_matrix, i, k), 
                                        sizeof(double) * N,
                                        getMatrixBlock(d_matrix_remote, i, k), 
                                        sizeof(double) * N, 
                                        sizeof(double) * B, 
                                        B, cudaMemcpyDeviceToDevice, s);
            checkCudaErrors(cublasDtrsm(
                cublasHandle,
                CUBLAS_SIDE_RIGHT, 
                CUBLAS_FILL_MODE_UPPER,
                CUBLAS_OP_N, 
                CUBLAS_DIAG_NON_UNIT, 
                B, B,
                &one,
                getMatrixBlock(d_matrix, k, k), N, // k + k * N;
                getMatrixBlock(d_matrix, i, k), N)); // (i + B) + k * N;
            if (copy) cudaMemcpy2DAsync(getMatrixBlock(d_matrix_remote, i, k), 
                                        sizeof(double) * N,
                                        getMatrixBlock(d_matrix, i, k), 
                                        sizeof(double) * N, 
                                        sizeof(double) * B, 
                                        B, cudaMemcpyDeviceToDevice, s);
            tiledLUGraphCreator->endCaptureOperation();

            for (int j = k + 1; j < 2; j++)
            {
                // A[j][i] = GEMM(A[j][k], A[i][k])
                // A[j][i] = A[j][i] - L[j][k] * L[i][k]^T
                // checkCudaErrors(cublasSetWorkspace(cublasHandle, d_workspace_cublas[(i-1)*T + (j-1)], workspaceInBytesOnDevice));
                tiledLUGraphCreator->beginCaptureOperation(
                    std::make_pair(i, j),
                    {std::make_pair(i, k), std::make_pair(k, j), std::make_pair(i, j)});
                if (copy) cudaMemcpy2DAsync(getMatrixBlock(d_matrix, i, k), 
                                            sizeof(double) * N,
                                            getMatrixBlock(d_matrix_remote, i, k), 
                                            sizeof(double) * N, 
                                            sizeof(double) * B, 
                                            B, cudaMemcpyDeviceToDevice, s);
                if (copy) cudaMemcpy2DAsync(getMatrixBlock(d_matrix, k, j), 
                                            sizeof(double) * N,
                                            getMatrixBlock(d_matrix_remote, k, j), 
                                            sizeof(double) * N, 
                                            sizeof(double) * B, 
                                            B, cudaMemcpyDeviceToDevice, s);
                // if (copy) copyMatrixBlock(d_matrix, d_matrix_remote, j, k, B);
                // if (copy) copyMatrixBlock(d_matrix, d_matrix_remote, i, k, B);
                // if (copy) copyMatrixBlock(d_matrix, d_matrix_remote, i, k, B);
                checkCudaErrors(cublasGemmEx(
                    cublasHandle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N, // CUBLAS_OP_T
                    B, B, B,
                    &minusOne,
                    getMatrixBlock(d_matrix, i, k), CUDA_R_64F, N, // i + k * N
                    getMatrixBlock(d_matrix, k, j), CUDA_R_64F, N, // j + i * N
                    &one,
                    getMatrixBlock(d_matrix, i, j), CUDA_R_64F, N, // k + i * N
                    CUBLAS_COMPUTE_64F,
                    CUBLAS_GEMM_DEFAULT));
                if (copy) cudaMemcpy2DAsync(getMatrixBlock(d_matrix_remote, i, j), 
                                            sizeof(double) * N,
                                            getMatrixBlock(d_matrix, i, j), 
                                            sizeof(double) * N, 
                                            sizeof(double) * B, 
                                            B, cudaMemcpyDeviceToDevice, s);
                tiledLUGraphCreator->endCaptureOperation();
            }
        }
    }

    checkCudaErrors(cudaDeviceSynchronize());
        
    cudaGraphExec_t graphExec;
    CudaEventClock clock;
    double totalTime;
    
    if (myPE == 0) {
        if (subgraph) {
            // checkCudaErrors(cudaGraphDebugDotPrint(graph, "./graph.dot", 0));
            if (verbose) showMemUsage();
            if (verbose) std::cout << "Initializing graphs..." << std::endl;

            cudaGraphExec_t *h_subgraphsExec = new cudaGraphExec_t[totalNodes];
            for (int i = 0; i < totalNodes; i++)
            {
                char filename[20];
                sprintf(filename, "./graph_%d.dot", i);
                if (dot)
                    checkCudaErrors(cudaGraphDebugDotPrint(tiledLUGraphCreator->subgraphs[i], filename, 0));
                checkCudaErrors(cudaGraphInstantiate(&h_subgraphsExec[i], tiledLUGraphCreator->subgraphs[i], 0));
            }

            checkCudaErrors(cudaDeviceSynchronize());
            if (verbose) showMemUsage();
            if (verbose) std::cout << "Launching..." << std::endl;
            
            for (int i = 0; i < runs; i++) {
                for (int j = 0; j < totalNodes; j++) {
                    clock.start(s);
                    checkCudaErrors(cudaGraphLaunch(h_subgraphsExec[j], s));
                    checkCudaErrors(cudaStreamSynchronize(s));
                    clock.end(s);
                    checkCudaErrors(cudaDeviceSynchronize());
                    float time = clock.getTimeInSeconds();
                    totalTime += time;
                    printf("subgraph %d | %d run | time (s): %4.4f\n", j, i, time);
                }
            }
            if (verbose) std::cout << "Done" << std::endl;
            
            free(h_subgraphsExec);
        } else {
            if (dot)
                checkCudaErrors(cudaGraphDebugDotPrint(graph, "./graph.dot", 0));
            checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
            for (int i = 0; i < runs; i++) {
                clock.start(s);
                checkCudaErrors(cudaGraphLaunch(graphExec, s));
                checkCudaErrors(cudaStreamSynchronize(s));
                clock.end(s);
                checkCudaErrors(cudaDeviceSynchronize());
                float time = clock.getTimeInSeconds();
                printf("device %d | %d run | time (s): %4.4f\n", myPE, i, time);
                totalTime += time;
            }
        }
        printf("Total time used (s): %4.4f\n", totalTime);
    }

    if (!subgraph) checkCudaErrors(cudaFree(d_matrix));
    else nvshmem_free(d_matrices);
    checkCudaErrors(cudaFree(d_info));
    // checkCudaErrors(cudaFree(h_workspace));
    checkCudaErrors(cudaFree(d_workspace_cusolver));
    for (int i = 0; i < workspaces; i++) {
        checkCudaErrors(cudaFree(d_workspace_cublas[i]));
    }
}

void LU(bool remote, bool copy, bool subgraph, bool dot)
{
    if (!remote) copy = false;
    tiledLU(remote, copy, subgraph, dot);
}

int main(int argc, char **argv)
{
    auto cmdl = argh::parser(argc, argv);

    if (!(cmdl({"N", "n"}, N) >> N)) {
        std::cerr << "Must provide a valid N value! Got '" << cmdl({"N", "n"}).str() << "'" << std::endl;
        return 0;
    }
    if (!(cmdl({"t", "T"}, T) >> T)) {
        std::cerr << "Must provide a valid T value! Got '" << cmdl({"T", "t"}).str() << "'" << std::endl;
        return 0;
    }
    if (N % T > 0) {
        std::cerr << "N must be divisible by T! Got 'N=" << N << " & T=" << T << "'" << std::endl;
        return 0;
    }
    if (!(cmdl({"sm", "SM", "smLimit"}, smLimit) >> smLimit) || smLimit > 108 || smLimit < 1) {
        std::cerr << "Must provide a valid SM Limit value! Got '" << cmdl({"sm", "SM", "smLimit"}).str() << "'" << std::endl;
        return 0;
    }
    if (!(cmdl({"workspace", "ws", "w", "W"}, workspace) >> workspace) || workspace > 1024*1024 || workspace < 1) {
        std::cerr << "Must provide a valid workspace (in kBytes) value! Got '" << cmdl({"workspace", "ws", "w"}).str() << "'" << std::endl;
        return 0;
    }
    if (!(cmdl({"run", "runs", "r", "R"}, runs) >> runs) || runs < 1) {
        std::cerr << "Must provide a valid number of runs! Got '" << cmdl({"run", "r", "R"}).str() << "'" << std::endl;
        return 0;
    }

    nvshmem_init();

    myPE = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);

    int gpusAvailable = -1;
    checkCudaErrors(cudaGetDeviceCount(&gpusAvailable));
    verbose = cmdl[{"v", "verbose"}] && myPE == 0;
    checkAndSetP2Paccess(nvshmem_n_pes());
    checkCudaErrors(cudaSetDevice(myPE));

    if (verbose) {
        printf("Hello from NVSHMEM_PE=%d/%d\n", myPE, nvshmem_n_pes());
        printf("%d GPUs detected, asked to use use %d GPUs\n", gpusAvailable, nvshmem_n_pes());
    }
    B = N / T;
    
    if (myPE == 0) {
        if (cmdl["subgraph"])
            std::cout << "SUBGRAPH";
        else 
            std::cout << "SINGLE GRAPH";
        if (cmdl["remote"])
            std::cout << " REMOTE";
        else 
            std::cout << " LOCAL";
        std::cout << " with N=" << N << " (" << T << " of " << B << "x" << B << " tiles)" << std::endl;

        if (cmdl["subgraph"]) {
            std::cout << "SM Limit per kernel = " << smLimit << std::endl;
            std::cout << "cuBLAS workspace = " << workspace << " kB" << std::endl;
        }
    }

    LU(cmdl["remote"], cmdl["copy"], cmdl["subgraph"], cmdl["dot"]);
    
    nvshmem_finalize();

    return 0;
}

// nvcc lu_partg_nvshmem.cu -I${NVSHMEM_PATH}/include -L${NVSHMEM_PATH}/lib -lcublas -lcusolver -lnvshmem -lnvidia-ml -o nvshmem_lu

// LOCAL
// subgraph 0 | 2 run | time (s): 0.0040
// subgraph 1 | 2 run | time (s): 0.0062
// subgraph 2 | 2 run | time (s): 0.0058
// subgraph 3 | 2 run | time (s): 0.0070

// REMOTE
// subgraph 0 | 1 run | time (s): 0.0725
// subgraph 1 | 1 run | time (s): 0.1266
// subgraph 2 | 1 run | time (s): 0.1402
// subgraph 3 | 1 run | time (s): 0.1385

// READ COPY 
// subgraph 0 | 0 run | time (s): 0.0057
// subgraph 1 | 0 run | time (s): 0.0082
// subgraph 2 | 0 run | time (s): 0.0077
// subgraph 3 | 0 run | time (s): 0.0104

// READ&WRITE COPY 
// subgraph 0 | 9 run | time (s): 0.0068
// subgraph 1 | 9 run | time (s): 0.0090
// subgraph 2 | 9 run | time (s): 0.0094
// subgraph 3 | 9 run | time (s): 0.0111