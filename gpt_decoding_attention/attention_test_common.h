#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include "hiprand_kernel.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#include "hip_utils.h"
#include "decoder_masked_multihead_attention.h"

#include "unfused_attention_kernels_ie_rocm.h"

typedef struct {
    int batch_size;
    int head_num;
    int max_seq_len;
    int max_output_len;
    int size_per_head;
    int rotary_dimension;
    int paged_block_size;
} test_args_t;

#define TIMEIT(print, n, ms, stream, fn, ...)                                                                              \
    ({                                                                                                                 \
        hipEvent_t _macro_event_start, _macro_event_stop;                                                             \
        hipEventCreate(&_macro_event_start);                                                                          \
        hipEventCreate(&_macro_event_stop);                                                                           \
        hipEventRecord(_macro_event_start, stream);                                                                   \
        for (int i = 0; i < n; i++) {                                                                                  \
            fn(__VA_ARGS__);                                                                                           \
        }                                                                                                              \
        hipEventRecord(_macro_event_stop, stream);                                                                    \
        hipStreamSynchronize(stream);                                                                                 \
        hipEventElapsedTime(&ms, _macro_event_start, _macro_event_stop);                                              \
        ms /= n;                                                                                                       \
        if (print)                                                                                                     \
            printf("[TIMEIT] " #fn ": %.2fµs\n", ms * 1000);                                                           \
        ms *= 1000;                                                                                                            \
    })


template<typename T>
struct rel_abs_diff {
    T operator()(const T& lhs, const T& rhs) const
    {
        return lhs == 0 ? 0 : static_cast<T>(fabs(lhs - rhs) / fabs(lhs));
    }
};

template<typename T>
struct abs_diff {
    T operator()(const T& lhs, const T& rhs) const
    {
        return static_cast<T>(fabs(lhs - rhs));
    }
};

template<typename T, template<typename W> class MMHA_PARAMS>
void set_params_struct(MMHA_PARAMS<T>& params,
                       T*                                    out,
                       const T*                              q,
                       const T*                              q_bias,
                       const T*                              k,
                       const T*                              k_bias,
                       const T*                              v,
                       const T*                              v_bias,
                       T*                                    k_cache,
                       T*                                    v_cache,
                       const int*                            cache_indir,
                       int                                   stride,
                       int                                   batch_size,
                       int                                   beam_width,
                       int                                   seq_length,
                       int                                   num_heads,
                       int                                   hidden_size_per_head,
                       int                                   rotary_embedding_dim,
                       int                                   timestep,
                       float                                 inv_sqrt_dh,
                       const int*                            input_lengths,
                       int                                   max_input_len,
                       const T*                              relative_attention_bias,
                       int                                   relative_attention_bias_stride,
                       int                                   paged_block_size,
                       int*                            cur_timesteps)
{
}

template<typename T>
void set_params_struct(Masked_multihead_attention_params<T>& params,
                       T*                                    out,
                       const T*                              q,
                       const T*                              q_bias,
                       const T*                              k,
                       const T*                              k_bias,
                       const T*                              v,
                       const T*                              v_bias,
                       T*                                    k_cache,
                       T*                                    v_cache,
                       const int*                            cache_indir,
                       int                                   stride,
                       int                                   batch_size,
                       int                                   beam_width,
                       int                                   seq_length,
                       int                                   num_heads,
                       int                                   hidden_size_per_head,
                       int                                   rotary_embedding_dim,
                       int                                   timestep,
                       float                                 inv_sqrt_dh,
                       const int*                            input_lengths,
                       int                                   max_input_len,
                       const T*                              relative_attention_bias,
                       int                                   relative_attention_bias_stride,
                       int                                   paged_block_size,
                       int*                            cur_timesteps)
{
    params.out                            = out;
    params.q                              = q;
    params.q_bias                         = q_bias;
    params.k                              = k;
    params.k_bias                         = k_bias;
    params.v                              = v;
    params.v_bias                         = v_bias;
    params.k_cache                        = k_cache;
    params.v_cache                        = v_cache;
    params.cache_indir                    = cache_indir;
    params.stride                         = stride;
    params.batch_size                     = batch_size;
    params.beam_width                     = beam_width;
    params.memory_max_len                 = seq_length;
    params.num_heads                      = num_heads;
    params.hidden_size_per_head           = hidden_size_per_head;
    params.rotary_embedding_dim           = rotary_embedding_dim;
    params.timestep                       = timestep;
    params.inv_sqrt_dh                    = inv_sqrt_dh;
    params.prefix_prompt_lengths          = input_lengths;
    params.max_input_length               = max_input_len;
    params.relative_attention_bias        = relative_attention_bias;
    params.relative_attention_bias_stride = relative_attention_bias_stride;

    params.neox_rotary_style = true;

    params.finished                 = nullptr;
    params.memory_length_per_sample = nullptr;
    params.length_per_sample        = nullptr;
}

template<typename T>
void set_params_struct(Paged_masked_multihead_attention_params<T>& params,
                       T*                                    out,
                       const T*                              q,
                       const T*                              q_bias,
                       const T*                              k,
                       const T*                              k_bias,
                       const T*                              v,
                       const T*                              v_bias,
                       T*                                    k_cache,
                       T*                                    v_cache,
                       const int*                            cache_indir,
                       int                                   stride,
                       int                                   batch_size,
                       int                                   beam_width,
                       int                                   seq_length,
                       int                                   num_heads,
                       int                                   hidden_size_per_head,
                       int                                   rotary_embedding_dim,
                       int                                   timestep,
                       float                                 inv_sqrt_dh,
                       const int*                            input_lengths,
                       int                                   max_input_len,
                       const T*                              relative_attention_bias,
                       int                                   relative_attention_bias_stride,
                       int                                   paged_block_size,
                       int*                            cur_timesteps)
{
    params.out                            = out;
    params.q                              = q;
    params.q_bias                         = q_bias;
    params.k                              = k;
    params.k_bias                         = k_bias;
    params.v                              = v;
    params.v_bias                         = v_bias;
    params.k_cache                        = k_cache;
    params.v_cache                        = v_cache;
    params.cache_indir                    = cache_indir;
    params.stride                         = stride;
    params.batch_size                     = batch_size;
    params.beam_width                     = beam_width;
    params.memory_max_len                 = seq_length;
    params.num_heads                      = num_heads;
    params.hidden_size_per_head           = hidden_size_per_head;
    params.rotary_embedding_dim           = rotary_embedding_dim;
    // params.timestep                       = timestep;
    params.inv_sqrt_dh                    = inv_sqrt_dh;
    params.prefix_prompt_lengths          = input_lengths;
    params.max_input_length               = max_input_len;
    params.relative_attention_bias        = relative_attention_bias;
    params.relative_attention_bias_stride = relative_attention_bias_stride;

    params.finished                 = nullptr;
    params.memory_length_per_sample = nullptr;
    params.length_per_sample        = nullptr;

    params.neox_rotary_style = true;
    
    params.tokens_per_block = paged_block_size;
    params.timestep = cur_timesteps;
    params.max_timestep = 5120;
    params.layer_index = 0;
}

template<typename T>
__global__ void cuda_random_uniform_kernel(T* buffer, const size_t size, const int seq_offset)
{
    const int     idx = blockIdx.x * blockDim.x + threadIdx.x;
    hiprandState_t local_state;
    hiprand_init((unsigned long long int)1337, idx + seq_offset, 0, &local_state);
    for (size_t index = idx; index < size; index += blockDim.x * gridDim.x) {
        buffer[index] = (T)(hiprand_uniform(&local_state) * 0.2f - 0.1f);
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
        buffer[index] = __float2bfloat16(round(hiprand_uniform(&local_state) * 10) - 5);
#endif
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

template<typename T_OUT, typename T_IN>
__global__ void cudaCast(T_OUT* dst, T_IN* src, const size_t size)
{
    for (size_t tid = threadIdx.x + blockIdx.x * blockDim.x; tid < size; tid += blockDim.x * gridDim.x) {
        dst[tid] = (T_OUT)((float)(src[tid]));
    }
}

template<typename T_OUT>
__global__ void cudaCast(T_OUT* dst, __nv_bfloat16 const* const src, const size_t size)
{
    for (size_t tid = threadIdx.x + blockIdx.x * blockDim.x; tid < size; tid += blockDim.x * gridDim.x) {
        dst[tid] = (T_OUT)(__bfloat162float(src[tid]));
    }
}

template<>
__global__ void cudaCast(__nv_bfloat16* dst, float const* const src, const size_t size)
{
    for (size_t tid = threadIdx.x + blockIdx.x * blockDim.x; tid < size; tid += blockDim.x * gridDim.x) {
        dst[tid] = __float2bfloat16(src[tid]);
    }
}

template<typename T_OUT, typename T_IN>
void invokeCudaCast(T_OUT* dst, T_IN const* const src, const size_t size, hipStream_t stream)
{
    cudaCast<<<256, 256, 0, stream>>>(dst, src, size);
}

template void invokeCudaCast(float* dst, half const* const src, const size_t size, hipStream_t stream);
template void invokeCudaCast(float* dst, __nv_bfloat16 const* const src, const size_t size, hipStream_t stream);

template<typename T>
class GPUBuf {
public:
    GPUBuf(size_t size, bool random_init = true): size(size), ptr(nullptr)
    {
        assert(size >= 0);
        check_cuda_error(hipMalloc((void**)(&ptr), sizeof(T) * size));
        if (random_init) {
            cudaRandomUniform(ptr, size);
        }
    }
    template<typename T2>
    GPUBuf(const GPUBuf<T2>& buf_src): size(buf_src.size), ptr(nullptr)
    {
        check_cuda_error(hipMalloc((void**)(&ptr), sizeof(T) * size));
        set(buf_src);
    }

    template<typename T2>
    void set(const GPUBuf<T2>& buf_src)
    {
        if (std::is_same<T, T2>::value) {
            check_cuda_error(hipMemcpy(ptr, reinterpret_cast<T*>(buf_src.ptr), sizeof(T) * size, hipMemcpyDeviceToDevice));
        }
        else {
            invokeCudaCast(ptr, buf_src.ptr, size, 0);
        }
    }

    void set(const T* h_ptr)
    {
        check_cuda_error(hipMemcpy(ptr, h_ptr, sizeof(T) * size, hipMemcpyHostToDevice));
    }

    std::vector<T> to_host_vec() const
    {
        std::vector<T> host_vec(size);
        check_cuda_error(hipMemcpy(host_vec.data(), ptr, sizeof(T) * size, hipMemcpyDeviceToHost));
        return host_vec;
    }

    ~GPUBuf() 
    {
        if (ptr != nullptr)
            hipFree(ptr);
    }

    size_t size;
    T*     ptr;

};

template<typename T>
GPUBuf<T> reshape_key_cache(const GPUBuf<T>& key_cache, int BS, int H, int Dh, int L, int x_orig, int x_targ)
{
    auto           h_key_cache = key_cache.to_host_vec();
    std::vector<T> h_key_cache_r(h_key_cache.size());

    for (int b = 0; b < BS; b++) {
        for (int h = 0; h < H; h++) {
            for (int d = 0; d < Dh; d++) {
                for (int l = 0; l < L; l++) {

                    int in_d = d / x_orig, out_d = d / x_targ;
                    int in_x = d % x_orig, out_x = d % x_targ;

                    int in_offset  = (((b * H + h) * (Dh / x_orig) + in_d) * L + l) * x_orig + in_x;
                    int out_offset = (((b * H + h) * (Dh / x_targ) + out_d) * L + l) * x_targ + out_x;

                    h_key_cache_r[out_offset] = h_key_cache[in_offset];
                }
            }
        }
    }
    GPUBuf<T> key_cache_T(key_cache.size);
    key_cache_T.set(h_key_cache_r.data());

    return key_cache_T;
}

__global__ void setPageBlockPtrs(size_t* dst, size_t src)
{
    size_t i = threadIdx.x;
    dst[i] = reinterpret_cast<size_t>(src);
}

template<typename T>
void invokeSetPageBlockPtrs(size_t* dst, const std::vector<T*> src, const size_t blocks)
{
    for(int i = 0; i < blocks; i++){
        setPageBlockPtrs<<<1, 1>>>(dst + i, reinterpret_cast<size_t>(src[i]));
    }
}

template<typename T>
void invokeSetBatchBlockPtrs(size_t* dst, const T* src, const size_t bs, const size_t stride)
{
    for(int i = 0; i < bs; i++){
        setPageBlockPtrs<<<1, 1>>>(dst + i, reinterpret_cast<size_t>(src + i * stride));
    }
}

template void invokeSetPageBlockPtrs(size_t* dst, const std::vector<float*> src, const size_t blocks);
template void invokeSetPageBlockPtrs(size_t* dst, const std::vector<half*> src, const size_t blocks);
template void invokeSetPageBlockPtrs(size_t* dst, const std::vector<__nv_bfloat16*> src, const size_t blocks);

template void invokeSetBatchBlockPtrs(size_t* dst, const float* src, const size_t bs, const size_t stride);
template void invokeSetBatchBlockPtrs(size_t* dst, const half* src, const size_t bs, const size_t stride);
template void invokeSetBatchBlockPtrs(size_t* dst, const __nv_bfloat16* src, const size_t bs, const size_t stride);

template<typename T>
struct string_rep_t {
    static const std::string value;
};
template<>
const std::string string_rep_t<half>::value{"FP16"};
template<>
const std::string string_rep_t<__nv_bfloat16>::value{"BF16"};

template<typename T>
struct mha_type_t {
    using Type = T;
};
template<>
struct mha_type_t<half> {
    using Type = uint16_t;
};

