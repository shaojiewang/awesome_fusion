#include "hip_utils.h"
#include "hip_type_utils.cuh"

template<typename T>
__global__ void transpose_4d_batch_major_k_cache_ptr(T***      k_dst,
                                                     const T*  k_src,
                                                     const int tokens_per_block,
                                                     const int layer_index,
                                                     const int head_num,
                                                     const int size_per_head,
                                                     const int seq_len,
                                                     const int max_seq_len)
{
    const int     batch_id = blockIdx.y;
    const int     head_id  = blockIdx.z;
    constexpr int X_ELEMS  = 16 / sizeof(T);
    auto      key_src = reinterpret_cast<const uint4*>(k_src + batch_id * head_num * size_per_head * seq_len
                                                  + head_id * size_per_head * seq_len);
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
        const int layer_stride = head_num * size_per_head * tokens_per_block;
        const int head_stride  = size_per_head * tokens_per_block;
        // get (layer_index, head_id, k_head_size_id, k_seq_len_id, 0) of k_cache
        auto key_dst = reinterpret_cast<uint4*>(
            &(k_dst[batch_id][k_seq_len_id / tokens_per_block][layer_index * layer_stride + head_id * head_stride
                                                               + k_head_size_id * tokens_per_block * X_ELEMS
                                                               + k_seq_len_id % tokens_per_block * X_ELEMS]));
        *key_dst = key_src[k_seq_len_id * size_per_head_div_x + k_head_size_id];
    }
}

template<typename T>
__global__ void transpose_4d_batch_major_v_cache_ptr(T***      v_dst,
                                                     const T*  v_src,
                                                     const int tokens_per_block,
                                                     const int layer_index,
                                                     const int head_num,
                                                     const int size_per_head,
                                                     const int seq_len,
                                                     const int max_seq_len)
{
    const int batch_id = blockIdx.y;
    const int head_id  = blockIdx.z;

    // 16 byte loads will handle "x" dimension
    auto val_src = reinterpret_cast<const uint4*>(v_src + batch_id * head_num * size_per_head * seq_len
                                                  + head_id * size_per_head * seq_len);

    // idx is over output dimension L * size_per_head / x for values
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    constexpr int X_ELEMS             = (sizeof(T) == 4) ? 4 : 8;
    const int     size_per_head_div_x = size_per_head / X_ELEMS;

    if (idx >= size_per_head_div_x * seq_len) {
        return;
    }

    const int v_seq_len_id   = idx / (size_per_head / X_ELEMS);
    const int v_head_size_id = idx % (size_per_head / X_ELEMS);
    const int layer_stride   = head_num * size_per_head * tokens_per_block;
    const int head_stride    = size_per_head * tokens_per_block;
    // get (layer_index, head_id, v_seq_len_id, v_head_size_id) of v_cache
    auto val_dst = reinterpret_cast<uint4*>(
        &(v_dst[batch_id][v_seq_len_id / tokens_per_block][layer_index * layer_stride + head_id * head_stride
                                                           + v_seq_len_id % tokens_per_block * size_per_head
                                                           + v_head_size_id * X_ELEMS]));
    *val_dst = val_src[idx];
}

template<typename T>
void invokeTranspose4dBatchMajorWithKVCachePtr(T***         k_dst,
                                               T***         v_dst,
                                               const T*     k_src,
                                               const T*     v_src,
                                               const int    tokens_per_block,
                                               const int    layer_index,
                                               const int    local_batch_size,
                                               const int    seq_len,
                                               const int    max_seq_len,
                                               const int    size_per_head,
                                               const int    local_head_num,
                                               hipStream_t stream)
{
    constexpr int block_sz = 128;
    constexpr int x        = 16 / sizeof(T);
    int           size     = max_seq_len * size_per_head / x;
    dim3          grid((size + block_sz - 1) / block_sz, local_batch_size, local_head_num);
    dim3          grid_v((seq_len * size_per_head / x + block_sz - 1) / block_sz, local_batch_size, local_head_num);

    transpose_4d_batch_major_k_cache_ptr<<<grid, block_sz, 0, stream>>>(
        k_dst, k_src, tokens_per_block, layer_index, local_head_num, size_per_head, seq_len, max_seq_len);
    sync_check_cuda_error();

    transpose_4d_batch_major_v_cache_ptr<<<grid_v, block_sz, 0, stream>>>(
        v_dst, v_src, tokens_per_block, layer_index, local_head_num, size_per_head, seq_len, max_seq_len);
    sync_check_cuda_error();
}

#define INSTANTIATETRANSPOSE4DBATCHMAJORWITHKVCACHEPTR(T)                                                              \
    template void invokeTranspose4dBatchMajorWithKVCachePtr(T***         k_dst,                                        \
                                                            T***         v_dst,                                        \
                                                            const T*     k_src,                                        \
                                                            const T*     v_src,                                        \
                                                            const int    tokens_per_block,                             \
                                                            const int    layer_index,                                  \
                                                            const int    local_batch_size,                             \
                                                            const int    seq_len,                                      \
                                                            const int    max_seq_len,                                  \
                                                            const int    size_per_head,                                \
                                                            const int    local_head_num,                               \
                                                            hipStream_t stream)
INSTANTIATETRANSPOSE4DBATCHMAJORWITHKVCACHEPTR(float);
INSTANTIATETRANSPOSE4DBATCHMAJORWITHKVCACHEPTR(half);
#ifdef ENABLE_BF16
INSTANTIATETRANSPOSE4DBATCHMAJORWITHKVCACHEPTR(__nv_bfloat16);
#endif
#undef INSTANTIATETRANSPOSE4DBATCHMAJORWITHKVCACHEPTR

