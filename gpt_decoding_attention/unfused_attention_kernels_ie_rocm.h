#pragma once

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
                                               cudaStream_t stream);

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
                                                    cudaStream_t stream);

template<typename T>
void invokeTranspose4dBatchMajor(T*           k_dst,
                                 T*           v_dst,
                                 const T*     k_src,
                                 const T*     v_src,
                                 const int    local_batch_size,
                                 const int    seq_len,
                                 const int    max_seq_len,
                                 const int    size_per_head,
                                 const int    local_head_num,
                                 hipStream_t stream);

