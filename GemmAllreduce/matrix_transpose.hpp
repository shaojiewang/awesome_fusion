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
    dim3 grid((n + 31) / 32, (k + 31) / 32);
    dim3 block(32, 32);
    matrix_transpose<<<grid, block, 0, stream>>>(dst, src, k, n);
}

template void invokeMatrixTranspose(float* dst, const float* src, const int m, const int n, hipStream_t stream);
template void invokeMatrixTranspose(half* dst, const half* src, const int m, const int n, hipStream_t stream);
template void invokeMatrixTranspose(hip_bfloat16* dst, const hip_bfloat16* src, const int m, const int n, hipStream_t stream);


// src is [bs, k, n] row-major
// dst is [bs, n, k] row-major
template<typename T>
__global__ void matrix_batched_transpose(T* dst, const T* src, const int k, const int n, const int bsz)
{
    __shared__ T shm[16][17];
    const int    tidx  = threadIdx.x;
    const int    tidy  = threadIdx.y;
    int          n_idx = blockIdx.x * 16 + tidx;
    int          k_idx = blockIdx.y * 16 + tidy;
    int          bsz_idx = blockIdx.z;
    if (n_idx < n && k_idx < k && bsz_idx < bsz) {
        shm[tidx][tidy] = src[bsz_idx * n * k + k_idx * n + n_idx];
    }

    __syncthreads();
    n_idx = blockIdx.x * 16 + tidy;
    k_idx = blockIdx.y * 16 + tidx;
    if (n_idx < n && k_idx < k && bsz_idx < bsz) {
        dst[bsz_idx * n * k + n_idx * k + k_idx] = shm[tidy][tidx];
    }
}

// src is [bs, k, n] row-major
// dst is [bs, n, k] row-major
template<>
__global__ void matrix_batched_transpose<uint8_t>(uint8_t* dst, const uint8_t* src, const int k, const int n, const int bsz)
{
    __shared__ uint8_t shm[16][17];
    const int    tidx  = threadIdx.x;
    const int    tidy  = threadIdx.y;
    int          n_idx = blockIdx.x * 16 + tidx;
    int          k_idx = blockIdx.y * 16 + tidy;
    int          bsz_idx = blockIdx.z;
    if (n_idx < n && k_idx < k && bsz_idx < bsz) {
        shm[tidx][tidy] = src[bsz_idx * n * k + k_idx * n + n_idx];
    }

    __syncthreads();
    n_idx = blockIdx.x * 16 + tidy;
    k_idx = blockIdx.y * 16 + tidx;
    if (n_idx < n && k_idx < k && bsz_idx < bsz) {
        dst[bsz_idx * n * k + n_idx * k + k_idx] = shm[tidy][tidx] - 128;
    }
}

// src is [k, n] row-major
// dst is [n, k] row-major
template<typename T>
void invokeMatrixBatchedTranspose(T* dst, const T* src, const int k, const int n, const int bsz, hipStream_t stream)
{
    dim3 grid((n + 15) / 16, (k + 15) / 16, bsz);
    dim3 block(16, 16);
    matrix_batched_transpose<<<grid, block, 0, stream>>>(dst, src, k, n, bsz);
}

template void invokeMatrixBatchedTranspose(float* dst, const float* src, const int m, const int n, const int bsz, hipStream_t stream);
template void invokeMatrixBatchedTranspose(half* dst, const half* src, const int m, const int n, const int bsz, hipStream_t stream);
template void invokeMatrixBatchedTranspose(hip_bfloat16* dst, const hip_bfloat16* src, const int m, const int n, const int bsz, hipStream_t stream);
template void invokeMatrixBatchedTranspose(uint8_t* dst, const uint8_t* src, const int m, const int n, const int bsz, hipStream_t stream);

}
