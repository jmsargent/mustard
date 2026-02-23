#!/bin/bash
# sargent@minerva:~/mustard/build_mustard.sh
# put cmake install in definition-file instead, also get python there
# right now requires manually downloaded cmake file to use

apptainer exec --nv container.sif bash -c "

    # 3. Clean and Patch
    rm -rf build && mkdir -p build
    
    # Apply patch to CMakeLists.txt (ensure we are in the root)
    sed -i 's|nvshmem::nvshmem|-lnvshmem_host -lnvshmem_device|g' CMakeLists.txt

    cd build

    # 4. Configure
    cmake .. \
        -DCMAKE_CXX_COMPILER=g++ \
        -DCMAKE_CUDA_COMPILER=\$CUDA_HOME/bin/nvcc \
        -DCMAKE_CUDA_ARCHITECTURES=89 \
        -DMUSTARD_CUDA_ARCHITECTURES=89 \
        -DCMAKE_EXE_LINKER_FLAGS=\"-L\$NVSHMEM_HOME/lib\" \
        -DCMAKE_CUDA_FLAGS=\"-I\$NVSHMEM_HOME/include -L\$NVSHMEM_HOME/lib -lnvshmem_host -lnvshmem_device -lcuda\"

    # 5. Build
    make -j\$(nproc)
"