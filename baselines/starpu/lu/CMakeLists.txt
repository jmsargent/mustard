find_package(PkgConfig)
pkg_check_modules(STARPU REQUIRED starpu-1.3)
if (STARPU_FOUND)
    # include_directories (${STARPU_INCLUDE_DIRS})
    message(STATUS ${STARPU_STATIC_LIBRARY_DIRS})
    # list(GET STARPU_STATIC_LIBRARY_DIRS 0 STARPU_DIR_ONLY)
    # set(STARPU_STATIC_LIBRARY_DIRS ${STARPU_DIR_ONLY})
    # message(STATUS ${STARPU_STATIC_LIBRARY_DIRS})
    # link_directories    (${STARPU_STATIC_LIBRARY_DIRS})
    # link_libraries      (${STARPU_STATIC_LIBRARIES})
else (STARPU_FOUND)
    message(FATAL_ERROR "StarPU not found")
endif()

set(STARPU_BLAS "-DSTARPU_OPENBLAS")
# set(BLA_VENDOR Intel10_64lp)
find_package( BLAS REQUIRED )
message( STATUS BLAS found: ${BLAS_LIBRARIES} )
# if (BLAS_FOUND)
#     set(STARPU_BLAS "-DSTARPU_MKL")
# endif()
message(STATUS ${STARPU_BLAS})

add_executable(lu_example_double lu_example_double.c dlu.c dlu_pivot.c dlu_kernels.c ../common/blas.c)

set(LIBS 
    CUDA::cudart
    CUDA::cublas
    CUDA::cusolver
    CUDA::cusparse
    CUDA::nvml)
message(STATUS ${LIBS})
message(STATUS ${STARPU_STATIC_LIBRARIES})
message(STATUS ${BLAS_LIBRARIES})


if (STARPU_FOUND)

    target_compile_options(lu_example_double PRIVATE ${STARPU_BLAS})

    # -I directories
    target_include_directories(lu_example_double PRIVATE ${STARPU_INCLUDE_DIRS})
 
    target_link_directories(lu_example_double PRIVATE ${STARPU_STATIC_LIBRARY_DIRS})

    # Static -l libs
    target_link_libraries(lu_example_double PUBLIC ${STARPU_STATIC_LIBRARIES} ${BLAS_LIBRARIES})
    # target_link_libraries(lu_example_double )
    # target_link_libraries(lu_example_double ${LIBS})
endif()

# target_compile_options(cholesky_partg PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
#         -D_FORCE_INLINES
#         >)

# target_compile_options(lu_partg PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
#         -D_FORCE_INLINES
#         >)

# target_include_directories(cholesky_partg SYSTEM PRIVATE 
#                             "${CMAKE_CURRENT_SOURCE_DIR}")