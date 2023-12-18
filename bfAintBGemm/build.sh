#!/bin/sh
SRC=bfAintBGemm.cc
OUT=bfAintBGemm
TOP=`pwd`
BUILD="$TOP/build/"

rm -rf $BUILD ; mkdir $BUILD ; cd $BUILD

/opt/rocm/bin/hipcc $TOP/$SRC -fPIC -DASM_PRINT -std=c++17 -O3 -Wall --offload-arch=gfx90a -save-temps -o $BUILD/$OUT

KSRC=bf16gemm_kernel_gfx90a.s
KOUT=bf16gemm_kernel_gfx90a.hsaco

/opt/rocm/llvm/bin/clang++ -x assembler -target amdgcn--amdhsa -mcpu=gfx90a $TOP/$KSRC -o $BUILD/$KOUT

