#!/bin/sh
ROCM_PATH=/opt/rocm/
SRC=test_decoder_attention.cc 
CU_SRC=decoder_masked_multihead_attention.cu 
OUT=test_decoder_attention
TOP=`pwd`
BUILD="$TOP/build/"

rm -rf $BUILD ; mkdir $BUILD ; cd $BUILD

# /opt/rocm/bin/hipcc $TOP/$SRC $TOP/$CU_SRC -fPIC -std=c++17 -O3 -Wall --offload-arch=gfx90a -save-temps -o $BUILD/$OUT

SRC=test_paged_decoder_attention.cc 
CU_SRC1=decoder_paged_masked_multihead_attention.cc 
OUT=test_paged_decoder_attention
SO_OUT=libpaged_decoder_attention.so

/opt/rocm/bin/hipcc $TOP/$SRC -DENABLE_BF16=1 -L../ -lhip_decoder_attn_lib -fPIC -std=c++17 -O3 -Wall -Wno-unused-result -Wno-unused-value --offload-arch=gfx90a -save-temps -o $BUILD/$OUT


