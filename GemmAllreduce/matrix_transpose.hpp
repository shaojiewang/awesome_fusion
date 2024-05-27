#pragma once

#include "hip_utils.h"
#include "hip_type_utils.cuh"

namespace awesome_fusion {

// src is [k, n] row-major
// dst is [n, k] row-major
template<typename T>
__global__ void matrix_transpose(T* dst, const T* src, const int k, const int n)
{
    __shared__ T shm[32][33];
    const int    tidx  = threadIdx.x;
    const int    tidy  = threadIdx.y;
    int          n_idx = blockIdx.x * 32 + tidx;
    int          k_idx = blockIdx.y * 32 + tidy;
    if (n_idx < n && k_idx < k) {
        shm[tidx][tidy] = src[k_idx * n + n_idx];
    }

    __syncthreads();
    n_idx = blockIdx.x * 32 + tidy;
    k_idx = blockIdx.y * 32 + tidx;
    if (n_idx < n && k_idx < k) {
        dst[n_idx * k + k_idx] = shm[tidy][tidx];
    }
}

// src is [k, n] row-major
// dst is [n, k] row-major
template<typename T>
void invokeMatrixTranspose(T* dst, const T* src, const int k, const int n, hipStream_t stream)
{
    dim3 grid(n / 32, k / 32);
    dim3 block(32, 32);
    matrix_transpose<<<grid, block, 0, stream>>>(dst, src, k, n);
}

template void invokeMatrixTranspose(float* dst, const float* src, const int m, const int n, hipStream_t stream);
template void invokeMatrixTranspose(half* dst, const half* src, const int m, const int n, hipStream_t stream);
template void invokeMatrixTranspose(hip_bfloat16* dst, const hip_bfloat16* src, const int m, const int n, hipStream_t stream);


}
