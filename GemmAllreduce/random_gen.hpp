#pragma once

#include <random>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hiprand/hiprand_kernel.h>

#include "hip_type_utils.cuh"
#include "hip_bf16_wrapper.hpp"

#define INT_INIT 1 

// cpu random gen
static inline void rand_vector_2d(float* v, int row, int col, int ld, float min_v = -1, float max_v = 1){
    int r, c;
    static int flag = 0;
    if(!flag){ srand(time(NULL)); flag = 1; }
    for(r = 0; r < row; r++){
        for(c = 0; c < col; c++){
            float tmp = float(std::rand()) / float(RAND_MAX);
            v[r * ld + c] = static_cast<float>(min_v + tmp * (max_v - min_v));
            // v[r * ld + c] = ((float)(r * ld + c)) / (row / 2 * col / 2) - 5;
        }
    }
}

static inline void rand_vector_2d_int_a(float* v, int row, int col, int ld){
    int r, c;
    static int flag = 0;
    if(!flag){ srand(time(NULL)); flag = 1; }
    for(r = 0; r < row; r++)
    {
        for(c = 0; c < col; c++)
        {
            v[r * ld + c] = ((float)(std::rand() % 4)) - 2;
            // v[r * ld + c] = (float)(r % 3 + 1);
            // v[r * ld + c] = 1; 
        }
    }
}

static inline void rand_vector_2d_int_b(float* v, int row, int col, int ld){
    int r, c;
    static int flag = 0;
    if(!flag){ srand(time(NULL)); flag = 1; }
    for(r = 0; r < row; r++)
    {
        for(c = 0; c < col; c++)
        {
            v[r * ld + c] = ((float)(std::rand() % 4)) - 2;
            // v[r * ld + c] = (float)(c % 3 + 1);
            // v[r * ld + c] = 1; 
        }
    }
}

static inline void rand_vector_2d_int_scale(float* v, int row, int col, int ld){
    int r, c;
    static int flag = 0;
    if(!flag){ srand(time(NULL)); flag = 1; }
    for(r = 0; r < row; r++)
    {
        for(c = 0; c < col; c++)
        {
            v[r * ld + c] = ((float)(std::rand() % 4)) - 2;
            v[r * ld + c] = (float)(c % 3 + 1);
        }
    }
}

static inline void rand_vector_2d_int(float* v, int row, int col, int ld){
    int r, c;
    static int flag = 0;
    if(!flag){ srand(time(NULL)); flag = 1; }
    for(r = 0; r < row; r++)
    {
        for(c = 0; c < col; c++)
        {
            v[r * ld + c] = ((float)(std::rand() % 4)) - 2;
            v[r * ld + c] = (float)(c % 3 + 1);
        }
    }
}

// GPU random gen
template<typename T>
__global__ void cuda_random_uniform_kernel(T* buffer, const size_t size, const int seq_offset)
{
    const int     idx = blockIdx.x * blockDim.x + threadIdx.x;
    hiprandState_t local_state;
    hiprand_init((unsigned long long int)1337, idx + seq_offset, 0, &local_state);
    for (size_t index = idx; index < size; index += blockDim.x * gridDim.x) {
#if !INT_INIT    
        buffer[index] = (T)(hiprand_uniform(&local_state) * 0.2f - 0.1f);
#else
        buffer[index] = (T)(round(hiprand_uniform(&local_state) * 2) - 1);
        buffer[index] = 1;
#endif
    }
}

template<>
__global__ void cuda_random_uniform_kernel<__nv_bfloat16>(__nv_bfloat16* buffer, const size_t size, const int seq_offset)
{
    const int     idx = blockIdx.x * blockDim.x + threadIdx.x;
    hiprandState_t local_state;
    hiprand_init((unsigned long long int)1337, idx + seq_offset, 0, &local_state);
    for (size_t index = idx; index < size; index += blockDim.x * gridDim.x) {
#if !INT_INIT    
        buffer[index] = __float2bfloat16(hiprand_uniform(&local_state) * 0.2f - 0.1f);
#else
        buffer[index] = __float2bfloat16(round(hiprand_uniform(&local_state) * 2) - 1);
        buffer[index] = __float2bfloat16(1.f);
#endif
    }
}

template<>
__global__ void cuda_random_uniform_kernel<int8_t>(int8_t* buffer, const size_t size, const int seq_offset)
{
    const int     idx = blockIdx.x * blockDim.x + threadIdx.x;
    hiprandState_t local_state;
    hiprand_init((float)1337.f, idx + seq_offset, 0, &local_state);
    for (size_t index = idx; index < size; index += blockDim.x * gridDim.x) {
        buffer[index] = hiprand(&local_state) % 0xFF;
        buffer[index] = 1; 
    }
}

template<typename T>
void cudaRandomUniform(T* buffer, const size_t size)
{
    static int seq_offset = 0;
    cuda_random_uniform_kernel<T><<<256, 256>>>(buffer, size, seq_offset);
    seq_offset += 256 * 256;
}

template void cudaRandomUniform(float* buffer, const size_t size);
template void cudaRandomUniform(half* buffer, const size_t size);
template void cudaRandomUniform(__nv_bfloat16* buffer, const size_t size);
template void cudaRandomUniform(int8_t* buffer, const size_t size);


