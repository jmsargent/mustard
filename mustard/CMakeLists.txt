add_executable(cholesky_partg cholesky_partg.cu)

add_executable(lu_partg lu_partg.cu)

find_package(NVSHMEM REQUIRED)
find_package(MPI REQUIRED)
find_package(METIS)

if(METIS_FOUND)
    target_link_libraries(cholesky_partg PUBLIC METIS::METIS)
    target_link_libraries(lu_partg PUBLIC METIS::METIS)
else()
    message(WARNING "METIS library not found. METIS support will be disabled.")
endif()

set(LIBS 
    CUDA::cudart
    CUDA::cublas
    CUDA::cusolver
    CUDA::nvml
    nvshmem::nvshmem
    MPI::MPI_CXX)

target_link_libraries(cholesky_partg ${LIBS})
target_link_libraries(lu_partg ${LIBS})

# target_compile_options(cholesky_partg PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
#         -D_FORCE_INLINES
#         >)

# target_compile_options(lu_partg PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
#         -D_FORCE_INLINES
#         >)

# target_include_directories(cholesky_partg SYSTEM PRIVATE 
#                             "${CMAKE_CURRENT_SOURCE_DIR}")