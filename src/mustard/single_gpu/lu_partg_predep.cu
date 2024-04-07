#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>
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

#include "../include/argh.h"
#include "broker_queue.h"
// #include "../utilities/cudaUtilities.hpp"

#define FLAGS_SUBG_COUNT 0

constexpr size_t N = 9 * 1;
constexpr size_t B = N / 3;
constexpr size_t T = N / B;


template <typename T>
void __check(T result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        fprintf(stderr, "CUDA error at %s:%d code=%d \"%s\" \n", file, line, static_cast<unsigned int>(result), func);
        exit(EXIT_FAILURE);
    }
}

#define checkCudaErrors(val) __check((val), #val, __FILE__, __LINE__)

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
    checkCudaErrors(cudaSetDevice(0));

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

bool verifyLUDecomposition(double *A, double *L, double *U, const int n)
{
    auto newA = std::make_unique<double[]>(n * n);
    memset(newA.get(), 0, n * n * sizeof(double));
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            for (int k = 0; k < n; k++)
            {
                newA[i * n + j] += L[i * n + k] * U[k * n + j];
            }
        }
    }

    double error = 0;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            error += fabs(A[i * n + j] - newA[i * n + j]);
        }
    }

    printf("A:\n");
    printSquareMatrix(A, n);

    printf("\nnewA:\n");
    printSquareMatrix(newA.get(), n);

    printf("\nL:\n");
    printSquareMatrix(L, n);
    printf("\n");

    printf("\nU:\n");
    printSquareMatrix(U, n);
    printf("\n");


    printf("error = %.6f}\n", error);

    return error <= 1e-6;
}

void trivialLU(bool verify)
{
    // Initialize libaries
    cusolverDnHandle_t cusolverDnHandle;
    checkCudaErrors(cusolverDnCreate(&cusolverDnHandle));

    cusolverDnParams_t cusolverDnParams;
    checkCudaErrors(cusolverDnCreateParams(&cusolverDnParams));

    // Initialize data
    double *h_A = (double *)malloc(N * N * sizeof(double));
    generateRandomSymmetricPositiveDefiniteMatrix(h_A, N);

    double *d_A;
    checkCudaErrors(cudaMalloc(&d_A, N * N * sizeof(double)));
    checkCudaErrors(cudaMemcpy(d_A, h_A, N * N * sizeof(double), cudaMemcpyHostToDevice));

    size_t workspaceInBytesOnDevice, workspaceInBytesOnHost;

    checkCudaErrors(cusolverDnXgetrf_bufferSize(
        cusolverDnHandle,
        cusolverDnParams,
        N,
        N,
        CUDA_R_64F,
        d_A,
        N,
        CUDA_R_64F,
        &workspaceInBytesOnDevice,
        &workspaceInBytesOnHost));

    void *h_workspace = malloc(workspaceInBytesOnHost);

    void *d_workspace;
    checkCudaErrors(cudaMalloc(&d_workspace, workspaceInBytesOnDevice));

    int *d_info;
    checkCudaErrors(cudaMalloc(&d_info, sizeof(int)));
    CudaEventClock clock;

    // Calculate
    clock.start();
    checkCudaErrors(cusolverDnXgetrf(
        cusolverDnHandle,
        cusolverDnParams,
        N,
        N,
        CUDA_R_64F,
        d_A,
        N,
        NULL, // no pivoting
        CUDA_R_64F,
        d_workspace,
        workspaceInBytesOnDevice,
        h_workspace,
        workspaceInBytesOnHost,
        d_info));
    clock.end();

    // Check
    int h_info = 0;
    checkCudaErrors(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0)
    {
        std::cout << "Unsuccessful potrf execution\n\n"
                  << "d_info = " << h_info << "\n\n";
    }

    // Verify
    if (verify) {
        double *h_L = (double *)malloc(N * N * sizeof(double));
        double *h_U = (double *)malloc(N * N * sizeof(double));
        checkCudaErrors(cudaMemcpy(h_L, d_A, N * N * sizeof(double), cudaMemcpyDeviceToHost));
        cleanCusolverLUDecompositionResult(h_L, h_U, N);
        printf("Result passes verification: %d\n", verifyLUDecomposition(h_A, h_L, h_U, N));

        // Clean
        free(h_L);
        free(h_U);
    }
    
    printf("Total time used (s): %4.4f\n", clock.getTimeInSeconds());

    free(h_A);
    free(h_workspace);
    checkCudaErrors(cusolverDnDestroy(cusolverDnHandle));
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_workspace));
    checkCudaErrors(cudaFree(d_info));
}

typedef std::pair<int, int> MatrixTile;

class TiledLUGraphCreator
{
public:
    cudaGraph_t *subgraphs;
    std::vector<std::vector<int>> subgraphDependencies;

    TiledLUGraphCreator(cudaStream_t stream, cudaGraph_t graph, bool subgraph = false, int totalNodes = 1) : stream(stream), graph(graph)
    {
        this->lastModifiedTile = std::make_pair(-1, -1);
        this->subgraph = subgraph;
        this->subgraphs = new cudaGraph_t[totalNodes];
        this->subgraphDependencies.resize(totalNodes);
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
            checkCudaErrors(cudaStreamBeginCaptureToGraph(this->stream, this->graph, dependencies.data(), nullptr, dependencies.size(), cudaStreamCaptureModeGlobal));
        } else {
            // auto dependencies = this->getSubgraphDependencies(tiles);
            this->subgraphDependencies[index_counter] = this->getSubgraphDependencies(tiles);
            checkCudaErrors(cudaGraphCreate(&this->subgraphs[index_counter], 0)); 
            std::cout << "Start capturing subgraph " << index_counter << std::endl;
            checkCudaErrors(cudaStreamBeginCaptureToGraph(this->stream, this->subgraphs[index_counter], nullptr, nullptr, 0, cudaStreamCaptureModeGlobal));
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
            std::cout << "End capturing subgraph " << index_counter << std::endl;
            this->index_counter++;
        }
        this->lastModifiedTile = std::make_pair(-1, -1);
    };
    
    void printDeps()
    {
        for (int i = 0; i < this->subgraphDependencies.size(); i++)
        {
            std::vector<int> deps = this->subgraphDependencies[i];
            std::cout << i << ":";
            for (int j = 0; j < deps.size(); j++)
            {
                std::cout << " " << deps[j];
            }
            std::cout << std::endl;
        }
    }

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

    std::vector<int> getSubgraphDependencies(std::vector<MatrixTile> tiles)
    {
        std::vector<int> dependencies;
        for (auto tile : tiles)
        {
            auto it = this->tileIndexByMap.find(tile);
            if (it != this->tileIndexByMap.end())
            {
                dependencies.push_back(it->second);
            }
        }

        auto dedupedEnd = std::unique(dependencies.begin(), dependencies.end());
        dependencies.resize(std::distance(dependencies.begin(), dedupedEnd));

        return dependencies;
    }

    cudaGraphNode_t getTailOfLastCapturedNodeChain()
    {
        if (lastDependencies.size() == 0)
        {
            size_t numEdges;
            checkCudaErrors(cudaGraphGetEdges(this->graph, nullptr, nullptr, &numEdges));
            auto from = std::make_unique<cudaGraphNode_t[]>(numEdges);
            auto to = std::make_unique<cudaGraphNode_t[]>(numEdges);
            checkCudaErrors(cudaGraphGetEdges(this->graph, from.get(), to.get(), &numEdges));

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

__global__ void kernel_populate_queue(BrokerWorkDistributor queue, int *dependencies, int totalNodes)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = tid; i < totalNodes; i = i + gridDim.x * blockDim.x)
    {
        if (dependencies[i] == 0) {
            // printf("Inserting %d", i);
            queue.enqueue(i);
        }
    }
}

__global__ void kernel_scheduler(
    BrokerWorkDistributor queue,
    volatile int *flags,
    cudaGraphExec_t *subgraphs,
    int totalSubgraphs,
    int device)
{
    unsigned int placeholder = UINT32_MAX;
    bool placeholder_bool = false;
    // int max_iter = 1000000;
    // int cur_iter = 0;

    printf("%d dev. Hello from scheduler starting with flags[FLAGS_SUBG_COUNT]=%d. Total subgraphs=%d\n", device, flags[0], totalSubgraphs);

    while (flags[FLAGS_SUBG_COUNT] < totalSubgraphs)
    //while (nvshmem_int_atomic_fetch((int *)&flags[FLAGS_SUBG_COUNT], 0) < totalSubgraphs)
    {
        if (/*flags[FLAGS_OCCUP] < (131072) && */queue.size() > 0)
        {
            queue.dequeue(placeholder_bool, placeholder); //, 0);
            // printf("%d dev. %d, flags[FLAGS_SUBG_COUNT]=%d\n", device, placeholder, nvshmem_int_atomic_fetch((int *)&flags[FLAGS_SUBG_COUNT], 0) );
            // for (int i = 0; i < 1000; i++) {
            //     __nanosleep(1000000U);
            // }
            cudaGraphLaunch(subgraphs[placeholder], cudaStreamGraphFireAndForget);
            //nvshmem_int_atomic_inc((int *)&flags[FLAGS_SUBG_COUNT], 0);
            atomicAdd((int *)&flags[FLAGS_SUBG_COUNT], 1);
            // printf("%d dev AFTER. %d, flags[FLAGS_SUBG_COUNT]=%d\n", device, placeholder, nvshmem_int_atomic_fetch((int *)&flags[FLAGS_SUBG_COUNT], 0) );
        }
        // cur_iter++;
    }

    // printf("%d dev. Exit from scheduler with flags[0]=%d\n", device, flags[FLAGS_SUBG_COUNT]);

    // if (flags[0] < 2) {
    //     printf("Scheduler self-relaunch with flags[0]=%d\n", all_flags[0][0]);
    //     // Query the current graph handle so we can relaunch it
    //     cudaGraphExec_t currentGraph = cudaGetCurrentGraphExec();
    //     cudaGraphLaunch(currentGraph, cudaStreamGraphTailLaunch);
    // }
}

void tiledLU(bool verify)
{
    // Initialize data
    auto originalMatrix = std::make_unique<double[]>(N * N); // Column-major
    generateRandomSymmetricPositiveDefiniteMatrix(originalMatrix.get(), N);

    // Copy to device
    double *d_matrix;
    checkCudaErrors(cudaMalloc(&d_matrix, N * N * sizeof(double)));
    checkCudaErrors(cudaMemcpy(d_matrix, originalMatrix.get(), N * N * sizeof(double), cudaMemcpyHostToDevice));

    auto getMatrixBlock = [&](int i, int j)
    {
        return d_matrix + i * B + j * B * N;
    };

    // Initialize libraries
    cusolverDnHandle_t cusolverDnHandle;
    cusolverDnParams_t cusolverDnParams;
    cublasHandle_t cublasHandle;
    checkCudaErrors(cusolverDnCreate(&cusolverDnHandle));
    checkCudaErrors(cusolverDnCreateParams(&cusolverDnParams));
    checkCudaErrors(cublasCreate(&cublasHandle));

    // Prepare constants
    double one = 1.0;
    double minusOne = -1.0;
    // double *one, *minusOne, *h_one, *h_minusOne;
    // checkCudaErrors(cudaMalloc(&one, sizeof(double)));
    // checkCudaErrors(cudaMalloc(&minusOne, sizeof(double)));
    // checkCudaErrors(cudaMallocHost(&h_one, sizeof(double)));
    // checkCudaErrors(cudaMallocHost(&h_minusOne, sizeof(double)));
    // *h_one = 1.0;
    // *h_minusOne = -1.0;
    // checkCudaErrors(cudaMemcpy((void *)one, (void *)h_one, 
    //                            sizeof(double), cudaMemcpyHostToDevice));
    // checkCudaErrors(cudaMemcpy((void *)minusOne, (void *)h_minusOne, 
    //                            sizeof(double), cudaMemcpyHostToDevice));

    // Prepare buffer for potrf
    size_t workspaceInBytesOnDevice, workspaceInBytesOnHost;
        
    checkCudaErrors(cusolverDnXgetrf_bufferSize(
        cusolverDnHandle,
        cusolverDnParams,
        B,
        B,
        CUDA_R_64F,
        d_matrix,
        N,
        CUDA_R_64F,
        &workspaceInBytesOnDevice,
        &workspaceInBytesOnHost));

    void *h_workspace, *d_workspace;
    int *d_info;
    checkCudaErrors(cudaMalloc(&h_workspace, workspaceInBytesOnHost));
    checkCudaErrors(cudaMalloc(&d_workspace, workspaceInBytesOnDevice));
    checkCudaErrors(cudaMalloc(&d_info, sizeof(int)));

    cudaGraph_t graph;
    checkCudaErrors(cudaGraphCreate(&graph, 0));

    int totalNodes = T;
    
    for (int k = 0; k < T; k++)
        for (int i = k + 1; i < T; i++)
            totalNodes += 2 + (T-(k+1));

    cudaStream_t s;
    checkCudaErrors(cudaStreamCreate(&s));

    checkCudaErrors(cusolverDnSetStream(cusolverDnHandle, s));
    checkCudaErrors(cublasSetStream(cublasHandle, s));
    checkCudaErrors(cublasSetWorkspace(cublasHandle, d_workspace, workspaceInBytesOnDevice));

    auto tiledLUGraphCreator = std::make_unique<TiledLUGraphCreator>(s, graph, true, totalNodes);

    size_t size_t_N = N;

    for (int k = 0; k < T; k++)
    {
        // A[k][k] = GETRF(A[k][k])
        // L[k][k]*U[k][k] = A[k][k]
        tiledLUGraphCreator->beginCaptureOperation(
            std::make_pair(k, k),
            {std::make_pair(k, k)});
        checkCudaErrors(cusolverDnXgetrf(
            cusolverDnHandle,
            cusolverDnParams,
            B,
            B,
            CUDA_R_64F,
            getMatrixBlock(k, k),
            N,
            NULL, // no pivoting
            CUDA_R_64F,
            d_workspace,
            workspaceInBytesOnDevice,
            h_workspace,
            workspaceInBytesOnHost,
            d_info));
        tiledLUGraphCreator->endCaptureOperation();

        for (int i = k + 1; i < T; i++)
        {
            // L[i][k] = TRSM(A[i][k], A[k][k]) // the U part of A[k][k]
            tiledLUGraphCreator->beginCaptureOperation(
                std::make_pair(k, i),
                {std::make_pair(k, k), std::make_pair(k, i)});
            checkCudaErrors(cublasDtrsm(
                cublasHandle,
                CUBLAS_SIDE_LEFT, // used to be right for cholesky
                CUBLAS_FILL_MODE_LOWER,
                CUBLAS_OP_N,// CUBLAS_OP_T for cholesky
                CUBLAS_DIAG_UNIT, // CUBLAS_DIAG_NON_UNIT for cholesky
                B, B,
                &one,
                getMatrixBlock(k, k), size_t_N, // k + k * N;
                getMatrixBlock(k, i), size_t_N)); // k + (i + B) * N;
            tiledLUGraphCreator->endCaptureOperation();

        }

        for (int i = k + 1; i < T; i++)
        {
            // U[k][i] = TRSM(A[k][k], A[k][i]) // the L part of A[k][k]
            tiledLUGraphCreator->beginCaptureOperation(
                std::make_pair(i, k),
                {std::make_pair(k, k), std::make_pair(i, k)});
            checkCudaErrors(cublasDtrsm(
                cublasHandle,
                CUBLAS_SIDE_RIGHT, 
                CUBLAS_FILL_MODE_UPPER,
                CUBLAS_OP_N, 
                CUBLAS_DIAG_NON_UNIT, 
                B, B,
                &one,
                getMatrixBlock(k, k), size_t_N, // k + k * N;
                getMatrixBlock(i, k), size_t_N)); // (i + B) + k * N;
            tiledLUGraphCreator->endCaptureOperation();

            for (int j = k + 1; j < T; j++)
            {
                // A[j][i] = GEMM(A[j][k], A[i][k])
                // A[j][i] = A[j][i] - L[j][k] * L[i][k]^T
                tiledLUGraphCreator->beginCaptureOperation(
                    std::make_pair(i, j),
                    {std::make_pair(i, k), std::make_pair(k, j), std::make_pair(i, j)});
                checkCudaErrors(cublasGemmEx(
                    cublasHandle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N, // CUBLAS_OP_T
                    B, B, B,
                    &minusOne,
                    getMatrixBlock(i, k), CUDA_R_64F, N, // i + k * N
                    getMatrixBlock(k, j), CUDA_R_64F, N, // j + i * N
                    &one,
                    getMatrixBlock(i, j), CUDA_R_64F, N, // k + i * N
                    CUBLAS_COMPUTE_64F,
                    CUBLAS_GEMM_DEFAULT));
                tiledLUGraphCreator->endCaptureOperation();
            }
        }
    }

    checkCudaErrors(cudaDeviceSynchronize());
    tiledLUGraphCreator->printDeps();
    
    volatile int *d_flags;
    int *d_dependencies, *h_dependencies;
    const int queue_size = totalNodes * 2;
    //BrokerWorkDistributor *queue = new BrokerWorkDistributor(queue_size);
    BrokerWorkDistributor queue(queue_size);

    checkCudaErrors(cudaMalloc(&d_flags, sizeof(int) * 32));
    checkCudaErrors(cudaMalloc(&d_dependencies, sizeof(int) * totalNodes));
    checkCudaErrors(cudaMallocHost(&h_dependencies, sizeof(int) * totalNodes));

    
    for (int i = 0; i < totalNodes; i++)
    {
        h_dependencies[i] = tiledLUGraphCreator->subgraphDependencies[i].size();
        std::cout << h_dependencies[i] << " ";
    }
    std::cout << std::endl;

    checkCudaErrors(cudaMemcpy((void *)d_dependencies, (void *)h_dependencies, 
                               sizeof(int) * totalNodes, cudaMemcpyHostToDevice));
    kernel_populate_queue<<<108, 1024>>>(queue, d_dependencies, totalNodes);
    // checkCudaErrors(cudaGraphDebugDotPrint(graph, "./graph.dot", 0));

    // cudaGraphExec_t graphExec;
    // checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    cudaGraphExec_t *h_subgraphsExec = new cudaGraphExec_t[totalNodes];
    cudaGraphExec_t *d_subgraphsExec;
    for (int i = 0; i < totalNodes; i++)
    {
        std::cout << i << std::endl;
        char filename[20];
        sprintf(filename, "./graph_%d.dot", i);
        checkCudaErrors(cudaGraphDebugDotPrint(tiledLUGraphCreator->subgraphs[i], filename, 0));
        checkCudaErrors(cudaGraphInstantiate(&h_subgraphsExec[i], tiledLUGraphCreator->subgraphs[i], cudaGraphInstantiateFlagDeviceLaunch));
        cudaGraphUpload(h_subgraphsExec[i], s);
    }
    checkCudaErrors(cudaMalloc(&d_subgraphsExec, sizeof(cudaGraphExec_t) * totalNodes));
    checkCudaErrors(cudaMemcpy((void *)d_subgraphsExec, (void *)h_subgraphsExec,
                                sizeof(cudaGraphExec_t) * 1, cudaMemcpyHostToDevice));

    cudaGraph_t schedulerGraph;
    cudaGraphExec_t schedulerExec;
    checkCudaErrors(cudaGraphCreate(&schedulerGraph, 0));
    cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
    kernel_scheduler<<<1, 1, 0, s>>>(queue, d_flags, d_subgraphsExec, 1, 0);
    cudaStreamEndCapture(s, &schedulerGraph);
    checkCudaErrors(cudaGraphInstantiate(&schedulerExec, schedulerGraph, cudaGraphInstantiateFlagDeviceLaunch));
    checkCudaErrors(cudaDeviceSynchronize());
    
    CudaEventClock clock;
    clock.start(s);
    // checkCudaErrors(cudaGraphLaunch(graphExec, s));
    checkCudaErrors(cudaGraphLaunch(schedulerExec, s));
    checkCudaErrors(cudaStreamSynchronize(s));
    clock.end(s);
    checkCudaErrors(cudaDeviceSynchronize());

    if (verify) {
        double *h_L = (double *)malloc(N * N * sizeof(double));
        double *h_U = (double *)malloc(N * N * sizeof(double));
        checkCudaErrors(cudaMemcpy(h_L, d_matrix, N * N * sizeof(double), cudaMemcpyDeviceToHost));
        cleanCusolverLUDecompositionResult(h_L, h_U, N);
        printf("Result passes verification: %d\n", verifyLUDecomposition(originalMatrix.get(), h_L, h_U, N));

        free(h_L);
        free(h_U);
    }
    printf("Total time used (s): %4.4f\n", clock.getTimeInSeconds());

    free(h_subgraphsExec);
    checkCudaErrors(cudaFreeHost(h_dependencies));
    checkCudaErrors(cudaFree(d_dependencies));
    checkCudaErrors(cudaFree((void*)d_flags));
    checkCudaErrors(cudaFree(d_subgraphsExec));
    checkCudaErrors(cudaFree(h_workspace));
    checkCudaErrors(cudaFree(d_workspace));
    checkCudaErrors(cudaFree(d_info));
    checkCudaErrors(cudaFree(d_matrix));
    queue.free_mem();
    // (*queue).free_mem();
    // delete queue;
}

void LU(bool tiled, bool verify)
{
    if (tiled)
    {
        tiledLU(verify);
    }
    else
    {
        trivialLU(verify);
    }
}

int main(int argc, char **argv)
{
    auto cmdl = argh::parser(argc, argv);

    LU(cmdl["tiled"], cmdl["verify"]);

    return 0;
}