#include "hip_utils.h"
#include "hip_type_utils.cuh"

#include "reduce_kernel_utils.cuh"
#include "decoder_masked_multihead_attention_utils.h"
#include "decoder_masked_multihead_attention_utils_ie.h"

template<typename T>
__global__ void transpose_4d_batch_major_k_cache_ptr(T*         kv_blocks,
                                                     size_t**   k_bt_offset,
                                                     const T*   k_src,
                                                     const int* input_lengths,
                                                     const int  tokens_per_block,
                                                     const int  layer_index,
                                                     const int  head_num,
                                                     const int  size_per_head,
                                                     const int  c_seq_len,
                                                     const int  max_seq_len)
{
    const int     batch_id = blockIdx.y;
    const int     head_id  = blockIdx.z;
    constexpr int X_ELEMS  = 16 / sizeof(T);
    const int     seq_len  = input_lengths[batch_id];

    auto      key_src = reinterpret_cast<const uint4*>(k_src + batch_id * head_num * size_per_head * c_seq_len
                                                  + head_id * size_per_head * c_seq_len);
    const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;

    int size_per_head_div_x = size_per_head / X_ELEMS;
    if (out_idx >= size_per_head_div_x * max_seq_len) {
        return;
    }

    int       idx            = out_idx;
    const int k_seq_len_id   = idx % max_seq_len;
    idx                      = (idx - k_seq_len_id) / max_seq_len;
    const int k_head_size_id = idx % size_per_head_div_x;

    if (k_seq_len_id < seq_len) {
        const size_t cur_k_bt_offset = k_bt_offset[batch_id][k_seq_len_id / tokens_per_block];
        const int    layer_stride    = head_num * size_per_head * tokens_per_block;
        const int    head_stride     = size_per_head * tokens_per_block;
        // get (layer_index, head_id, k_head_size_id, k_seq_len_id, 0) of k_cache
        auto key_dst = reinterpret_cast<uint4*>(
            &(kv_blocks[cur_k_bt_offset + layer_index * layer_stride + head_id * head_stride
                        + k_head_size_id * tokens_per_block * X_ELEMS + k_seq_len_id % tokens_per_block * X_ELEMS]));
        *key_dst = key_src[k_seq_len_id * size_per_head_div_x + k_head_size_id];
    }
}

template<typename T>
__global__ void transpose_4d_batch_major_v_cache_ptr(T*         kv_blocks,
                                                     size_t**   v_bt_offset,
                                                     const T*   v_src,
                                                     const int* input_lengths,
                                                     const int  tokens_per_block,
                                                     const int  layer_index,
                                                     const int  head_num,
                                                     const int  size_per_head,
                                                     const int  c_seq_len,
                                                     const int  max_seq_len)
{
    const int batch_id = blockIdx.y;
    const int head_id  = blockIdx.z;
    const int seq_len  = input_lengths[batch_id];

    // 16 byte loads will handle "x" dimension
    auto val_src = reinterpret_cast<const uint4*>(v_src + batch_id * head_num * size_per_head * c_seq_len
                                                  + head_id * size_per_head * c_seq_len);

    // idx is over output dimension L * size_per_head / x for values
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    constexpr int X_ELEMS             = (sizeof(T) == 4) ? 4 : 8;
    const int     size_per_head_div_x = size_per_head / X_ELEMS;

    if (idx >= size_per_head_div_x * c_seq_len) {
        return;
    }

    const int v_seq_len_id = idx / (size_per_head / X_ELEMS);
    if (v_seq_len_id < seq_len) {
        const int    v_head_size_id  = idx % (size_per_head / X_ELEMS);
        const int    layer_stride    = head_num * size_per_head * tokens_per_block;
        const int    head_stride     = size_per_head * tokens_per_block;
        const size_t cur_v_bt_offset = v_bt_offset[batch_id][v_seq_len_id / tokens_per_block];
        // get (layer_index, head_id, v_seq_len_id, v_head_size_id) of v_cache
        auto val_dst = reinterpret_cast<uint4*>(
            &(kv_blocks[cur_v_bt_offset + layer_index * layer_stride + head_id * head_stride
                        + v_seq_len_id % tokens_per_block * size_per_head + v_head_size_id * X_ELEMS]));
        *val_dst = val_src[idx];
    }
}

template<typename T>
void invokeTranspose4dBatchMajorWithKVCachePtr(T*           kv_blocks,
                                               size_t**     k_bt_offset,
                                               size_t**     v_bt_offset,
                                               const T*     k_src,
                                               const T*     v_src,
                                               const int*   input_lengths,
                                               const int    tokens_per_block,
                                               const int    layer_index,
                                               const int    local_batch_size,
                                               const int    c_seq_len,
                                               const int    max_seq_len,
                                               const int    size_per_head,
                                               const int    local_head_num,
                                               cudaStream_t stream)
{
    constexpr int block_sz = 128;
    constexpr int x        = 16 / sizeof(T);
    int           size     = max_seq_len * size_per_head / x;
    dim3          grid((size + block_sz - 1) / block_sz, local_batch_size, local_head_num);
    dim3          grid_v((c_seq_len * size_per_head / x + block_sz - 1) / block_sz, local_batch_size, local_head_num);

    transpose_4d_batch_major_k_cache_ptr<<<grid, block_sz, 0, stream>>>(kv_blocks,
                                                                        k_bt_offset,
                                                                        k_src,
                                                                        input_lengths,
                                                                        tokens_per_block,
                                                                        layer_index,
                                                                        local_head_num,
                                                                        size_per_head,
                                                                        c_seq_len,
                                                                        max_seq_len);
    sync_check_cuda_error();

    transpose_4d_batch_major_v_cache_ptr<<<grid_v, block_sz, 0, stream>>>(kv_blocks,
                                                                          v_bt_offset,
                                                                          v_src,
                                                                          input_lengths,
                                                                          tokens_per_block,
                                                                          layer_index,
                                                                          local_head_num,
                                                                          size_per_head,
                                                                          c_seq_len,
                                                                          max_seq_len);
    sync_check_cuda_error();
}

#define INSTANTIATETRANSPOSE4DBATCHMAJORWITHKVCACHEPTR(T)                                                              \
    template void invokeTranspose4dBatchMajorWithKVCachePtr(T*           kv_blocks,                                    \
                                                            size_t**     k_bt_offset,                                  \
                                                            size_t**     v_bt_offset,                                  \
                                                            const T*     k_src,                                        \
                                                            const T*     v_src,                                        \
                                                            const int*   input_lengths,                                \
                                                            const int    tokens_per_block,                             \
                                                            const int    layer_index,                                  \
                                                            const int    local_batch_size,                             \
                                                            const int    c_seq_len,                                    \
                                                            const int    max_seq_len,                                  \
                                                            const int    size_per_head,                                \
                                                            const int    local_head_num,                               \
                                                            cudaStream_t stream)
INSTANTIATETRANSPOSE4DBATCHMAJORWITHKVCACHEPTR(float);
INSTANTIATETRANSPOSE4DBATCHMAJORWITHKVCACHEPTR(half);
#ifdef ENABLE_BF16
INSTANTIATETRANSPOSE4DBATCHMAJORWITHKVCACHEPTR(__nv_bfloat16);
#endif
#undef INSTANTIATETRANSPOSE4DBATCHMAJORWITHKVCACHEPTR

template<typename KV_T, typename T>
__global__ void transpose_4d_batch_major_k_cache_ptr_quant(KV_T*      kv_blocks,
                                                           size_t**   k_bt_offset,
                                                           const T*   k_src,
                                                           float**    k_scale_orig_quant,
                                                           const int* input_lengths,
                                                           const int  tokens_per_block,
                                                           const int  layer_index,
                                                           const int  head_num,
                                                           const int  size_per_head,
                                                           const int  c_seq_len,
                                                           const int  max_seq_len)
{
    const int     seq_id              = blockIdx.x;
    const int     batch_id            = blockIdx.y;
    const int     head_id             = blockIdx.z;
    const int     seq_len             = input_lengths[batch_id];
    constexpr int X_ELEMS             = 16 / sizeof(T);
    int           size_per_head_div_x = size_per_head / X_ELEMS;
    using T_dst                       = KV_T;
    using T_src                       = typename mmha::packed_type<T, X_ELEMS>::type;
    using T_scale                     = typename mmha::kv_cache_scale_type_t<T, KV_T>::Type;
    __shared__ float s_max, scale_f;
    float            local_max = -FLT_MAX;

    if (seq_id >= seq_len) {
        return;
    }

    auto key_src = reinterpret_cast<const T_src*>(k_src + batch_id * head_num * size_per_head * c_seq_len
                                                  + head_id * size_per_head * c_seq_len + seq_id * size_per_head);

    const size_t cur_k_bt_offset = k_bt_offset[batch_id][seq_id / tokens_per_block];
    const int    layer_stride    = head_num * size_per_head * tokens_per_block;
    const int    head_stride     = size_per_head * tokens_per_block;
    // get (layer_index, head_id, tid, seq_id, 0) of k_cache
    auto key_dst =
        reinterpret_cast<T_dst*>(&(kv_blocks[cur_k_bt_offset + layer_index * layer_stride + head_id * head_stride]));
    auto k_scale_orig_quant_ = k_scale_orig_quant[batch_id] + head_id * max_seq_len;

    for (int tid = threadIdx.x; tid < size_per_head_div_x; tid += blockDim.x) {
        float tid_max = mmha::fabs_max(key_src[tid]);
        if (local_max < tid_max)
            local_max = tid_max;
    }

    float max_val = blockDim.x <= 32 ? warpReduceMax(local_max) : blockReduceMax(local_max);
    if (threadIdx.x == 0) {
        s_max                       = max_val;
        scale_f                     = 127 / s_max;
        k_scale_orig_quant_[seq_id] = 1.0 / scale_f;
    }
    __syncthreads();

    for (int tid = threadIdx.x; tid < size_per_head_div_x; tid += blockDim.x) {
        const int channelIdx = tid * tokens_per_block + seq_id % tokens_per_block;
        int       inBlockIdx = channelIdx * sizeof(typename mmha::packed_type<T_dst, mmha::num_elems<T_src>::value>::type);
        T_src     val        = key_src[tid];

        // Cast float scale to dst data type.
        T_scale scaleOrigQuant;
        mmha::convert_from_float(&scaleOrigQuant, scale_f);

        // Store 8bits kv cache.
        mmha::store_8bits_kv_cache_vec(key_dst, val, inBlockIdx, scaleOrigQuant);
    }
}

template<typename KV_T, typename T>
__global__ void transpose_4d_batch_major_v_cache_ptr_quant(KV_T*      kv_blocks,
                                                           size_t**   v_bt_offset,
                                                           const T*   v_src,
                                                           float**    v_scale_orig_quant,
                                                           const int* input_lengths,
                                                           const int  tokens_per_block,
                                                           const int  layer_index,
                                                           const int  head_num,
                                                           const int  size_per_head,
                                                           const int  c_seq_len,
                                                           const int  max_seq_len)
{
    const int tokenIdx = blockIdx.x;
    const int batch_id = blockIdx.y;
    const int head_id  = blockIdx.z;

    const int seq_len = input_lengths[batch_id];
    // We allow only fp32/fp16/bf16 as input types
    static_assert(sizeof(T) == 4 || sizeof(T) == 2, "");
    constexpr int X_ELEMS             = (sizeof(T) == 4) ? 4 : 8;
    const int     size_per_head_div_x = size_per_head / X_ELEMS;
    using T_dst                       = KV_T;
    using T_src                       = typename mmha::packed_type<T, X_ELEMS>::type;
    using T_scale                     = typename mmha::kv_cache_scale_type_t<T, KV_T>::Type;

    float            local_max = -FLT_MAX;
    __shared__ float s_max, scale_f;

    if (tokenIdx >= seq_len) {
        return;
    }

    // 16 byte loads will handle "x" dimension
    auto val_src = reinterpret_cast<const T_src*>(v_src + batch_id * head_num * size_per_head * c_seq_len
                                                  + head_id * size_per_head * c_seq_len + tokenIdx * size_per_head);

    const int    v_seq_len_id    = tokenIdx;
    const int    v_head_size_id  = threadIdx.x;
    const int    layer_stride    = head_num * size_per_head * tokens_per_block;
    const int    head_stride     = size_per_head * tokens_per_block;
    const size_t cur_v_bt_offset = v_bt_offset[batch_id][v_seq_len_id / tokens_per_block];
    // get (layer_index, head_id, v_seq_len_id, v_head_size_id) of v_cache
    auto val_dst =
        reinterpret_cast<T_dst*>(&(kv_blocks[cur_v_bt_offset + layer_index * layer_stride + head_id * head_stride
                                             + v_seq_len_id % tokens_per_block * size_per_head]));

    auto v_scale_orig_quant_ = v_scale_orig_quant[batch_id] + head_id * max_seq_len;

    for (int tid = threadIdx.x; tid < size_per_head_div_x; tid += blockDim.x) {
        float tid_max = mmha::fabs_max(val_src[tid]);
        if (local_max < tid_max)
            local_max = tid_max;
    }
    float max_val = blockDim.x <= 32 ? warpReduceMax(local_max) : blockReduceMax(local_max);
    if (threadIdx.x == 0) {
        s_max                         = max_val;
        scale_f                       = 127.0 / s_max;
        v_scale_orig_quant_[tokenIdx] = 1.0 / scale_f;
    }
    __syncthreads();

    if (threadIdx.x >= size_per_head_div_x) {
        return;
    }

    T_src val = val_src[threadIdx.x];
    // const int channelIdx = v_head_size_id;
    int inBlockIdx = v_head_size_id * sizeof(typename mmha::packed_type<T_dst, mmha::num_elems<T_src>::value>::type);

    // Cast float scale to dst data type.
    T_scale scaleOrigQuant;
    mmha::convert_from_float(&scaleOrigQuant, scale_f);

    // Store 8bits kv cache.
    mmha::store_8bits_kv_cache_vec(val_dst, val, inBlockIdx, scaleOrigQuant);
}

template<typename KV_T, typename T>
void invokeTranspose4dBatchMajorWithKVCachePtrQuant(KV_T*        kv_blocks,
                                                    size_t**     k_bt_offset,
                                                    size_t**     v_bt_offset,
                                                    float**      k_scale_orig_quant,
                                                    float**      v_scale_orig_quant,
                                                    const T*     k_src,
                                                    const T*     v_src,
                                                    const int*   input_lengths,
                                                    const int    tokens_per_block,
                                                    const int    layer_index,
                                                    const int    local_batch_size,
                                                    const int    c_seq_len,
                                                    const int    max_seq_len,
                                                    const int    size_per_head,
                                                    const int    local_head_num,
                                                    cudaStream_t stream)
{
    constexpr int x        = 16 / sizeof(T);
    const int     block_sz = min(1024, (size_per_head / x + 31) / 32 * 32);

    dim3 grid(c_seq_len, local_batch_size, local_head_num);
    dim3 grid_v(c_seq_len, local_batch_size, local_head_num);

    transpose_4d_batch_major_k_cache_ptr_quant<<<grid, block_sz, 0, stream>>>(kv_blocks,
                                                                              k_bt_offset,
                                                                              k_src,
                                                                              k_scale_orig_quant,
                                                                              input_lengths,
                                                                              tokens_per_block,
                                                                              layer_index,
                                                                              local_head_num,
                                                                              size_per_head,
                                                                              c_seq_len,
                                                                              max_seq_len);
    sync_check_cuda_error();

    transpose_4d_batch_major_v_cache_ptr_quant<<<grid_v, block_sz, 0, stream>>>(kv_blocks,
                                                                                v_bt_offset,
                                                                                v_src,
                                                                                v_scale_orig_quant,
                                                                                input_lengths,
                                                                                tokens_per_block,
                                                                                layer_index,
                                                                                local_head_num,
                                                                                size_per_head,
                                                                                c_seq_len,
                                                                                max_seq_len);
    sync_check_cuda_error();
}

#define INSTANTIATETRANSPOSE4DBATCHMAJORWITHKVCACHEPTRQUANT(KV_T, T)                                                   \
    template void invokeTranspose4dBatchMajorWithKVCachePtrQuant(KV_T*        kv_blocks,                               \
                                                                 size_t**     k_bt_offset,                             \
                                                                 size_t**     v_bt_offset,                             \
                                                                 float**      k_scale_orig_quant,                      \
                                                                 float**      v_scale_orig_quant,                      \
                                                                 const T*     k_src,                                   \
                                                                 const T*     v_src,                                   \
                                                                 const int*   input_lengths,                           \
                                                                 const int    tokens_per_block,                        \
                                                                 const int    layer_index,                             \
                                                                 const int    local_batch_size,                        \
                                                                 const int    c_seq_len,                               \
                                                                 const int    max_seq_len,                             \
                                                                 const int    size_per_head,                           \
                                                                 const int    local_head_num,                          \
                                                                 cudaStream_t stream)
INSTANTIATETRANSPOSE4DBATCHMAJORWITHKVCACHEPTRQUANT(int8_t, float);
INSTANTIATETRANSPOSE4DBATCHMAJORWITHKVCACHEPTRQUANT(int8_t, half);
#ifdef ENABLE_BF16
INSTANTIATETRANSPOSE4DBATCHMAJORWITHKVCACHEPTRQUANT(int8_t, __nv_bfloat16);
#endif
#undef INSTANTIATETRANSPOSE4DBATCHMAJORWITHKVCACHEPTRQUANT

