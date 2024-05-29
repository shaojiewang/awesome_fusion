#!/bin/bash
ROCM_PATH=/opt/rocm/
SRC=gemmAR.cc
# CU_SRC=decoder_masked_multihead_attention.cu 
OUT=test_gemm_ar
TOP=`pwd`
BUILD="$TOP/build/"

rm -rf $BUILD ; mkdir $BUILD ; cd $BUILD

# ${ROCM_PATH}/llvm/bin/clang++ 
hipcc ${TOP}/custom_ar_kernels.cu ${TOP}/custom_ar_comm.cc ${TOP}/nccl.cc ${TOP}/${SRC} -I/usr/local/include -pthread -L/usr/local/lib -L /opt/rocm/rccl/lib/ -Wl,-rpath -Wl,/usr/local/lib -Wl,--enable-new-dtags -lmpi -lrccl -lrocblas -std=c++17 -DBUILD_MULTI_GPU=ON --offload-arch=gfx90a -o ${BUILD}/${OUT} --save-temps
# mpirun -np 4 --allow-run-as-root build/gemm_ar_test

KSRC=bf16gemm_rr16r_b256_32x128x64_wg1x1_w1x4_32x32x8bf16_1k_pregld1_pipeline_interleaved_splitk.s
KOUT=bf16gemm_rr16r_b256_32x128x64_wg1x1_w1x4_32x32x8bf16_1k_pregld1_pipeline_interleaved_splitk.hsaco

/opt/rocm/llvm/bin/clang++ -x assembler -target amdgcn--amdhsa -mcpu=gfx90a $TOP/$KSRC -o $BUILD/$KOUT

