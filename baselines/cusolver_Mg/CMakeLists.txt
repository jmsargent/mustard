add_executable(cusolver_MgGetrf_example cusolver_MgGetrf_example.cu)

add_executable(cusolver_MgPotrf_example1 cusolver_MgPotrf_example1.cu)

set(LIBS 
    CUDA::cudart)    

# This part is adapted from https://github.com/Kitware/CMake/blob/master/Modules/FindCUDA.cmake"
macro(cuda_find_library_local_first_with_path_ext _var _names _doc _path_ext )
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    # CUDA 3.2+ on Windows moved the library directories, so we need the new
    # and old paths.
    set(_cuda_64bit_lib_dir "${_path_ext}lib/x64" "${_path_ext}lib64" "${_path_ext}libx64" )
endif()
# CUDA 3.2+ on Windows moved the library directories, so we need to new
# (lib/Win32) and the old path (lib).
find_library(${_var}
    NAMES ${_names}
    PATHS "${CUDA_TOOLKIT_TARGET_DIR}" "${CUDA_TOOLKIT_ROOT_DIR}"
    ENV CUDA_HOME
    ENV LD_LIBRARY_PATH
    PATH_SUFFIXES ${_cuda_64bit_lib_dir} "${_path_ext}lib/Win32" "${_path_ext}lib" "${_path_ext}libWin32"
    DOC ${_doc}
    NO_DEFAULT_PATH
    )
# Search default search paths, after we search our own set of paths.
find_library(${_var}
    NAMES ${_names}
    PATHS "/usr/lib/nvidia-current"
    DOC ${_doc}
    )
endmacro()

macro(cuda_find_library_local_first _var _names _doc)
    cuda_find_library_local_first_with_path_ext( "${_var}" "${_names}" "${_doc}" "" )
endmacro()

macro(find_library_local_first _var _names _doc )
    cuda_find_library_local_first( "${_var}" "${_names}" "${_doc}" "" )
endmacro()

macro(FIND_CUDA_HELPER_LIBS _name)
    cuda_find_library_local_first(CUDA_${_name}_LIBRARY ${_name} "\"${_name}\" library")
    mark_as_advanced(CUDA_${_name}_LIBRARY)
endmacro()

find_cuda_helper_libs(cusolverMg)
set(LIBS "${LIBS} ${CUDA_cusolverMg_LIBRARY}")

target_link_libraries(cusolver_MgGetrf_example ${CUDA_cusolverMg_LIBRARY})
target_link_libraries(cusolver_MgPotrf_example1 ${CUDA_cusolverMg_LIBRARY})

# target_include_directories(cholesky_partg SYSTEM PRIVATE 
#                             "${CMAKE_CURRENT_SOURCE_DIR}")
