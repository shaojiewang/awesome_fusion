#include <stdio.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <random>
#include <iostream>
#include <stdlib.h>
#include <math.h>

#include "simple_device_mem.hpp"
#include "gemm_tensor_layout.hpp"

using Row = gemm_layout::gemm::RowMajor;
using Col = gemm_layout::gemm::ColMajor;


int main(int argc, char ** argv)
{
    int validation = 0;
    int m = 3840;
    int n = 4096;
    int k = 4096;
    if(argc >= 2) {
        validation = atoi(argv[1]);
    }
    if(argc >= 5) {
        m = atoi(argv[2]);
        n = atoi(argv[3]);
        k = atoi(argv[4]);
    }
    int lda = k;
    int ldb = k;
    int ldc = n;

    if(argc >= 8) {
        lda = atoi(argv[5]);
        ldb = atoi(argv[6]);
        ldc = atoi(argv[7]);
    }

    float *host_a, *host_b, *host_c;
    float16 *fp16_a, *fp16_b, *fp16_c, *dev_a, *dev_b, *dev_c;

    auto f_matrix_space_size = 
        [](std::size_t nRow, std::size_t nCol, std::size_t stride, auto layout){
            using Layout = decltype(layout);
            if constexpr(std::is_same<Layout, Row>::value) {
                return (nRow - 1) * stride + nCol;
            } else {
                return (nCol - 1) * stride + nRow;
            }
        };

    SimpleDeviceMem a_device_buf();
}
