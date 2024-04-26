#!/bin/bash
ROCM_PATH=/opt/rocm/
SRC=gemmAR.cc
# CU_SRC=decoder_masked_multihead_attention.cu 
OUT=test_gemm_ar
TOP=`pwd`
BUILD="$TOP/build/"

rm -rf $BUILD ; mkdir $BUILD ; cd $BUILD

# ${ROCM_PATH}/llvm/bin/clang++ 
hipcc ${TOP}/custom_ar_kernels.cu ${TOP}/custom_ar_comm.cc ${TOP}/nccl.cc ${TOP}/${SRC} -I/usr/local/include -pthread -L/usr/local/lib -L /opt/rocm/rccl/lib/ -Wl,-rpath -Wl,/usr/local/lib -Wl,--enable-new-dtags -lmpi -lrccl -std=c++14 -DBUILD_MULTI_GPU=ON --offload-arch=gfx90a -o ${BUILD}/${OUT} --save-temps
# mpirun -np 4 --allow-run-as-root build/gemm_ar_test

