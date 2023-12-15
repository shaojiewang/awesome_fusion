#include <stdio.h>
#include <hip/hip_runtime.h>
#include <random>
#include <iostream>
#include <stdlib.h>
#include <math.h>

#include "simple_device_mem.hpp"
#include "gemm_tensor_layout.hpp"
#include "datatype.hpp"
#include "host_ref.hpp"
#include "validation.hpp"
#include "random_gen.hpp"

using Row = gemm_layout::gemm::RowMajor;
using Col = gemm_layout::gemm::ColumnMajor;

using ALayout = Row;
using BLayout = Row;
using ScaleLayout = Row;
using CLayout = Row;

using ADataType = bfloat16;
using BDataType = int8_t;
using ScaleDataType = float;
using CDataType = bfloat16;



int main(int argc, char ** argv)
{
    int validation = 0;
    int m = 32;
    int n = 64;
    int k = 128;
    if(argc >= 2) {
        validation = atoi(argv[1]);
    }
    if(argc >= 5) {
        m = atoi(argv[2]);
        n = atoi(argv[3]);
        k = atoi(argv[4]);
    }
    int lda = k;
    int ldb = n;
    int ldc = n;

    if(argc >= 8) {
        lda = atoi(argv[5]);
        ldb = atoi(argv[6]);
        ldc = atoi(argv[7]);
    }

    auto f_matrix_space_size = 
        [](std::size_t nRow, std::size_t nCol, std::size_t stride, auto layout){
            using Layout = decltype(layout);
            if constexpr(std::is_same<Layout, Row>::value) {
                return (nRow - 1) * stride + nCol;
            } else {
                return (nCol - 1) * stride + nRow;
            }
        };

    SimpleDeviceMem a_device_buf(sizeof(ADataType) * f_matrix_space_size(m, k, lda, ALayout{}));
    SimpleDeviceMem b_device_buf(sizeof(BDataType) * f_matrix_space_size(k, n, ldb, BLayout{}));
    SimpleDeviceMem c_device_buf(sizeof(CDataType) * f_matrix_space_size(m, n, ldc, CLayout{}));
    SimpleDeviceMem scale_device_buf(sizeof(ScaleDataType) * f_matrix_space_size(n, 1, 1, ScaleLayout{}));
    
    SimpleHostMem a_host_buf(sizeof(float) * f_matrix_space_size(m, k, lda, ALayout{}));
    SimpleHostMem b_host_buf(sizeof(float) * f_matrix_space_size(k, n, ldb, BLayout{}));
    SimpleHostMem c_host_buf(sizeof(float) * f_matrix_space_size(m, n, ldc, CLayout{}));
    SimpleHostMem scale_host_buf(sizeof(float) * f_matrix_space_size(n, 1, 1, ScaleLayout{}));

    rand_vector_2d_int(reinterpret_cast<float*>(a_host_buf.GetBuffer()), m, k, lda);
    rand_vector_2d_int(reinterpret_cast<float*>(b_host_buf.GetBuffer()), n, k, ldb);
    rand_vector_2d_int(reinterpret_cast<float*>(scale_host_buf.GetBuffer()), n, k, ldb);

    SimpleHostMem a_host_buf_to_device(sizeof(ADataType) * f_matrix_space_size(m, k, lda, ALayout{}));
    SimpleHostMem b_host_buf_to_device(sizeof(BDataType) * f_matrix_space_size(k, n, ldb, BLayout{}));
    SimpleHostMem c_host_buf_from_device(sizeof(CDataType) * f_matrix_space_size(m, n, ldc, CLayout{}));


}
