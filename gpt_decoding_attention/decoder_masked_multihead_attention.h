/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <hip/hip_fp16.h>
#include <hip/hip_runtime_api.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef ENABLE_BF16
#include "hip_bf16_wrapper.h"
#endif

#define MAX_SEQLEN_TILE 64

////////////////////////////////////////////////////////////////////////////////////////////////////

#define CHECK_CUDA(call)                                                                                               \
    do {                                                                                                               \
        hipError_t status_ = call;                                                                                    \
        if (status_ != hipSuccess) {                                                                                  \
            fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, hipGetErrorString(status_));              \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)

////////////////////////////////////////////////////////////////////////////////////////////////////

// The structure of parameters for the masked multihead attention kernel.
//
// We use the following terminology to describe the different dimensions.
//
// B:  Batch size (number of sequences),
// L:  Sequence length,
// D:  Hidden dimension,
// H:  Number of heads,
// Dh: Hidden dimension per head - Dh = D / H.

template<typename T, bool SPLIT_KV_CACHE>
struct KVCacheType {
};

template<>
struct KVCacheType<float, false> {
    using Type     = float;
    using StepType = int;
};

template<>
struct KVCacheType<float, true> {
    using Type     = float;
    using StepType = int*;
};

template<>
struct KVCacheType<uint16_t, false> {
    using Type     = uint16_t;
    using StepType = int;
};

template<>
struct KVCacheType<uint16_t, true> {
    using Type     = uint16_t;
    using StepType = int*;
};

template<>
struct KVCacheType<int8_t, false> {
    using Type     = int8_t;
    using StepType = int;
};

template<>
struct KVCacheType<int8_t, true> {
    using Type     = int8_t;
    using StepType = int*;
};

#ifdef ENABLE_BF16
template<>
struct KVCacheType<__nv_bfloat16, false> {
    using Type     = __nv_bfloat16;
    using StepType = int;
};

template<>
struct KVCacheType<__nv_bfloat16, true> {
    using Type     = __nv_bfloat16;
    using StepType = int*;
};
#endif

template<typename T>
struct Multihead_attention_params_base {

    // The output buffer. Dimensions B x D.
    T* out = nullptr;

    // The input Qs and the associated bias. Dimensions B x D and D, resp.
    const T *q = nullptr, *q_bias = nullptr;
    // The input Ks and the associated bias. Dimensions B x D and D, resp.
    const T *k = nullptr, *k_bias = nullptr;
    // The input Vs and the associated bias. Dimensions B x D and D, resp.
    const T *v = nullptr, *v_bias = nullptr;

    // The cache for the Ks. The size must be at least B x L x D.
    T* k_cache = nullptr;
    // The cache for the Vs. The size must be at least B x L x D.
    T* v_cache = nullptr;
    // The indirections to use for cache when beam sampling.
    const int* cache_indir = nullptr;

    // scales
    const float* query_weight_output_scale               = nullptr;
    const float* attention_qk_scale                      = nullptr;
    const float* attention_output_weight_input_scale_inv = nullptr;

    // Stride to handle the case when KQV is a single buffer
    int stride = 0;

    // The batch size.
    int batch_size = 0;
    // The beam width
    int beam_width = 0;
    // The sequence length.
    int memory_max_len = 0;
    // The number of heads (H).
    int num_heads = 0;
    // The hidden dimension per head (Dh).
    int hidden_size_per_head = 0;
    // The per-head latent space reserved for rotary embeddings.
    int  rotary_embedding_dim = 0;
    bool neox_rotary_style    = false;
    // The maximum length of input sentences.
    int max_input_length = 0;
    // The current timestep. TODO(bhsueh) Check that do we only this param in cross attention?
    int timestep = 0;
    // The current timestep of each sentences (support different timestep for different sentences)

    // The 1.f / sqrt(Dh). Computed on the host.
    float inv_sqrt_dh = 0.0f;

    // Used when we have some input context like gpt
    const int* total_padding_tokens = nullptr;

    const bool* masked_tokens            = nullptr;
    const int*  prefix_prompt_lengths    = nullptr;
    int         max_prefix_prompt_length = 0;

    const T* relative_attention_bias        = nullptr;
    int      relative_attention_bias_stride = 0;
    // The slope per head of linear position bias to attention score (H).
    const float* linear_bias_slopes = nullptr;

    const T*   ia3_key_weights   = nullptr;
    const T*   ia3_value_weights = nullptr;
    const int* ia3_tasks         = nullptr;

    const float* qkv_scale_out       = nullptr;
    const float* attention_out_scale = nullptr;
    int          int8_mode           = 0;

    int          kv_cache_quant_mode = 0;
    float**      k_scale_cache_ptr   = nullptr;
    float**      v_scale_cache_ptr   = nullptr;

    int heads_per_gqa_group = 1;
    float rope_theta = 10000.0f;

    // Multi-block setups
    mutable bool enable_multi_block = false;

    // Number of streaming processors on the device.
    // Tune block size to maximum occupancy.
    int multi_processor_count = 104 * 2;

    mutable int timesteps_per_block        = -1;
    mutable int seq_len_tile               = -1;
    mutable int max_seq_len_tile           = MAX_SEQLEN_TILE;
    mutable int max_timesteps_per_block    = 8192;

    // The partial output buffer. Dimensions max_seq_len_tile x B x D. (for each timestep only seq_len_tile x B x D is
    // needed)
    T* partial_out = nullptr;
    // ThreadBlock sum. Dimensions max_seq_len_tile x 1. (for each timestep only seq_len_tile x 1 is needed)
    float* partial_sum = nullptr;
    // ThreadBlock max. Dimensions max_seq_len_tile x 1. (for each timestep only seq_len_tile x 1 is needed)
    float* partial_max = nullptr;
    // threadblock counter to identify the complete of partial attention computations
    int* block_counter = nullptr;

    const int* memory_length_per_sample = nullptr;
};

template<typename T, bool CROSS_ATTENTION, bool SPLIT_KV_CACHE>
struct Paged_multihead_attention_params: public Multihead_attention_params_base<T> {
    using KV_CACHE_T = typename KVCacheType<T, SPLIT_KV_CACHE>::Type;
    using STEP_T     = typename KVCacheType<T, SPLIT_KV_CACHE>::StepType;

    // Base ptr of key/value cache block
    KV_CACHE_T* kv_blocks = nullptr;
    // The cache for the Ks. The size must be at least B x L x D.
    size_t** k_cache = nullptr;
    // The cache for the Vs. The size must be at least B x L x D.
    size_t** v_cache = nullptr;

    // Number of tokens in each block of KV cache.
    int tokens_per_block;
    int layer_index;
    // The maximum timestep of input sentences
    int max_timestep = 0;
    // The current timestep of each sentences (support different timestep for different sentences)
    STEP_T timestep = 0;

    // output cross attentions
    float* cross_attention_out        = nullptr;
    int    max_decoder_seq_len        = 0;
    bool   is_return_cross_attentions = false;

    // allows to exist attention eary
    bool* finished = nullptr;

    // required in case of cross attention
    // will need it here till if constexpr in c++17
    int* memory_length_per_sample = nullptr;

    // required in case of masked attention with different length
    const int* length_per_sample = nullptr;
};

template<typename T, bool CROSS_ATTENTION>
struct Multihead_attention_params: public Multihead_attention_params_base<T> {
    // output cross attentions
    float* cross_attention_out        = nullptr;
    int    max_decoder_seq_len        = 0;
    bool   is_return_cross_attentions = false;

    // allows to exist attention eary
    bool* finished = nullptr;

    // required in case of cross attention
    // will need it here till if constexpr in c++17
    int* memory_length_per_sample = nullptr;

    // required in case of masked attention with different length
    const int* length_per_sample = nullptr;
};

template<typename T>
struct Multihead_attention_params<T, true>: public Multihead_attention_params_base<T> {
    // output cross attentions
    float* cross_attention_out        = nullptr;
    int    max_decoder_seq_len        = 0;
    bool   is_return_cross_attentions = false;

    // allows to exist attention eary
    bool* finished = nullptr;

    // required in case of cross attention
    int* memory_length_per_sample = nullptr;

    // required in case of masked attention with different length
    const int* length_per_sample = nullptr;
};

template<class T>
using Masked_multihead_attention_params = Multihead_attention_params<T, false>;

template<class T>
using Cross_multihead_attention_params = Multihead_attention_params<T, true>;

template<class T>
using Paged_masked_multihead_attention_params = Paged_multihead_attention_params<T, false, true>;

template<typename T>
struct outputCrossAttentionParam {
    // max decoder output length
    int  max_decoder_seq_len        = 0;
    T*   cross_attention_out        = nullptr;
    bool is_return_cross_attentions = false;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

void masked_multihead_attention(const Masked_multihead_attention_params<float>& params, const hipStream_t& stream);
void masked_multihead_attention(const Masked_multihead_attention_params<uint16_t>& params, const hipStream_t& stream);
#ifdef ENABLE_BF16
void masked_multihead_attention(const Masked_multihead_attention_params<__nv_bfloat16>& params,
                                const hipStream_t&                                     stream);
#endif
#ifdef ENABLE_FP8
void masked_multihead_attention(const Masked_multihead_attention_params<__nv_fp8_e4m3>& params,
                                const hipStream_t&                                     stream);
#endif
void cross_multihead_attention(const Cross_multihead_attention_params<float>& params, const hipStream_t& stream);
void cross_multihead_attention(const Cross_multihead_attention_params<uint16_t>& params, const hipStream_t& stream);
#ifdef ENABLE_BF16
void cross_multihead_attention(const Cross_multihead_attention_params<__nv_bfloat16>& params,
                               const hipStream_t&                                    stream);
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////
// paged attention function interface
void paged_masked_multihead_attention(const Paged_masked_multihead_attention_params<float>& params, const hipStream_t& stream);
void paged_masked_multihead_attention(const Paged_masked_multihead_attention_params<uint16_t>& params, const hipStream_t& stream);
#ifdef ENABLE_BF16
void paged_masked_multihead_attention(const Paged_masked_multihead_attention_params<__nv_bfloat16>& params, const hipStream_t& stream);
#endif



