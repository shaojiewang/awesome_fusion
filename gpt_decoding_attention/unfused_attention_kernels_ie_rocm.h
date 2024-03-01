#pragma once

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
                                               hipStream_t stream);

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

