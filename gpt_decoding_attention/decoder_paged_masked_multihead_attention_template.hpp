#pragma once

#include "decoder_masked_multihead_attention_template.hpp"

namespace mmha {
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, bool DO_CROSS_ATTENTION = false, bool SPLIT_KV_CACHE = true>
inline size_t smem_size_in_bytes(const Paged_multihead_attention_params<T, DO_CROSS_ATTENTION, SPLIT_KV_CACHE>& params,
                                 int threads_per_value,
                                 int threads_per_block)
{
    using Tk = typename kernel_type_t<T>::Type;
    // The amount of shared memory needed to store the Q*K^T values in float.
    const int max_timesteps = min(params.max_timestep, params.memory_max_len);
    size_t qk_sz = (DO_CROSS_ATTENTION) ? div_up(params.memory_max_len + 1, 4) * 16 : div_up(max_timesteps + 1, 4) * 16;

    // The extra memory needed if we are not using floats for the final logits.
    size_t logits_sz = 0;
#ifndef MMHA_USE_FP32_ACUM_FOR_LOGITS
    if (sizeof(Tk) != 4) {
        // TDOD
        logits_sz = (DO_CROSS_ATTENTION) ? div_up(params.memory_max_len + 1, 4) * 4 * sizeof(Tk) :
                                           div_up(max_timesteps + 1, 4) * 4 * sizeof(Tk);
    }
#endif

    // The total size needed during softmax.
    size_t softmax_sz = qk_sz + logits_sz;

    // The number of partial rows to reduce in the final reduction.
    int rows_per_red = threads_per_block / threads_per_value;
    // The amount of storage needed to finalize the outputs.
    size_t red_sz = rows_per_red * params.hidden_size_per_head * sizeof(Tk) / 2;

    size_t transpose_rotary_size = 0;
    if (params.rotary_embedding_dim > 0 && params.neox_rotary_style) {
        transpose_rotary_size = 2 * params.rotary_embedding_dim * sizeof(Tk);
    }

    // The max.
    return max(max(softmax_sz, red_sz), transpose_rotary_size);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, typename KV_CACHE_T, bool SPLIT_KV_CACHE>
inline __device__ T* get_kv_cache_ptr(KV_CACHE_T* kv_cache_base,
                                      const int   batch_index,
                                      const int   layer_index,
                                      const int   head_index,
                                      const int   token_index,
                                      const int   tokens_per_block,
                                      const int   layer_stride,
                                      const int   head_stride,
                                      const int   head_offset);

#define GET_KV_CACHE_PTR0(T1, T2)                                                                                      \
    template<>                                                                                                         \
    inline __device__ T1* get_kv_cache_ptr<T1, T2, false>(T2 * kv_cache_base,                                          \
                                                          const int batch_index,                                       \
                                                          const int layer_index,                                       \
                                                          const int head_index,                                        \
                                                          const int token_index,                                       \
                                                          const int tokens_per_block,                                  \
                                                          const int layer_stride,                                      \
                                                          const int head_stride,                                       \
                                                          const int head_offset)                                       \
    {                                                                                                                  \
        assert(false);                                                                                                 \
    }

#define GET_KV_CACHE_PTR1(T1, T2)                                                                                      \
    template<>                                                                                                         \
    inline __device__ T1* get_kv_cache_ptr<T1, T2, true>(T2 * kv_cache_base,                                           \
                                                         const int batch_index,                                        \
                                                         const int layer_index,                                        \
                                                         const int head_index,                                         \
                                                         const int token_index,                                        \
                                                         const int tokens_per_block,                                   \
                                                         const int layer_stride,                                       \
                                                         const int head_stride,                                        \
                                                         const int head_offset)                                        \
    {                                                                                                                  \
        T2 batch_kvcache_base = kv_cache_base[batch_index]; \
        T1* block_kvcache_base = batch_kvcache_base[token_index / tokens_per_block]; \ 
        return block_kvcache_base + (layer_index * layer_stride + head_index * head_stride + head_offset); \
    }
        //return &(kv_cache_base[batch_index][token_index / tokens_per_block]                                            \
                              [layer_index * layer_stride + head_index * head_stride + head_offset]);                  \
    }

GET_KV_CACHE_PTR0(float, float)
GET_KV_CACHE_PTR0(uint16_t, uint16_t)
GET_KV_CACHE_PTR0(int8_t, int8_t)
#ifdef ENABLE_BF16
GET_KV_CACHE_PTR0(__nv_bfloat16, __nv_bfloat16)
#endif
GET_KV_CACHE_PTR1(float, float**)
GET_KV_CACHE_PTR1(uint16_t, uint16_t**)
GET_KV_CACHE_PTR1(int8_t, int8_t**)
#ifdef ENABLE_BF16
GET_KV_CACHE_PTR1(__nv_bfloat16, __nv_bfloat16**)
#endif

#undef GET_KV_CACHE_PTR0
#undef GET_KV_CACHE_PTR1

template<typename STEP_T, bool SPLIT_KV_CACHE>
inline __device__ int get_cur_timestep(STEP_T timestep, const int bbi);

template<>
inline __device__ int get_cur_timestep<int, false>(int timestep, const int bbi)
{
    return timestep;
}

template<>
inline __device__ int get_cur_timestep<int*, true>(int* timestep, const int bbi)
{
    return timestep[bbi] - 1;
}

template<
    // The type of the inputs. Supported types: float and half.
    typename T,
    // The type of the k/v cache.
    typename Tcache,
    // The hidden dimension per head.
    int Dh,
    int Dh_MAX,
    // The number of threads per key.
    int THREADS_PER_KEY,
    // The number of threads per value.
    int THREADS_PER_VALUE,
    // The number of threads in a threadblock.
    int  THREADS_PER_BLOCK,
    bool DO_CROSS_ATTENTION,
    bool HAS_BEAMS,
    bool SPLIT_KV_CACHE = false>
__global__ void
paged_masked_multihead_attention_kernel(Paged_multihead_attention_params<T, DO_CROSS_ATTENTION, SPLIT_KV_CACHE> params)
{
    static constexpr bool ENABLE_8BITS_CACHE = sizeof(Tcache) == 1;

    using KV_CACHE_T = typename KVCacheType<Tcache, SPLIT_KV_CACHE>::Type;
    using STEP_T     = typename KVCacheType<Tcache, SPLIT_KV_CACHE>::StepType;
    using Tk         = typename kernel_type_t<T>::Type;
#ifdef ENABLE_FP8
    // FP8 MHA Scales
    constexpr bool FP8_MHA_KERNEL = std::is_same<T, __nv_fp8_e4m3>::value;
#else
    constexpr bool FP8_MHA_KERNEL = false;
#endif

    KV_CACHE_T* inp_kcache = reinterpret_cast<KV_CACHE_T*>(params.k_cache);
    KV_CACHE_T* inp_vcache = reinterpret_cast<KV_CACHE_T*>(params.v_cache);

    // Make sure the hidden dimension per head is a multiple of the number of threads per key.
    static_assert(Dh_MAX % THREADS_PER_KEY == 0, "");
    // Make sure the hidden dimension per head is a multiple of the number of threads per value.
    static_assert(Dh_MAX % THREADS_PER_VALUE == 0, "");

    // The size of a warp.
    constexpr int WARP_SIZE = 32;
    // The number of warps in a threadblock.
    constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / WARP_SIZE;

    // Use smem_size_in_bytes (above) to determine the amount of shared memory.
    extern __shared__ char smem_[];

    // The shared memory for the Q*K^T values and partial logits in softmax.
    float* qk_smem = reinterpret_cast<float*>(smem_);

    // The shared memory for the logits. For FP32, that's the same buffer as qk_smem.
    char* logits_smem_ = smem_;
#ifndef MMHA_USE_FP32_ACUM_FOR_LOGITS
    if (sizeof(Tk) != 4) {
        // TODO - change to tlength
        const int max_timesteps = min(params.max_timestep, params.memory_max_len);
        logits_smem_ +=
            (DO_CROSS_ATTENTION) ? div_up(params.memory_max_len + 1, 4) * 16 : div_up(max_timesteps + 1, 4) * 16;
    }
    Tk* logits_smem = reinterpret_cast<Tk*>(logits_smem_);
#else
    float*         logits_smem    = reinterpret_cast<float*>(logits_smem_);
#endif

    // The shared memory to do the final reduction for the output values. Reuse qk_smem.
    Tk* out_smem = reinterpret_cast<Tk*>(smem_);

    // The shared memory buffers for the block-wide reductions. One for max, one for sum.
    __shared__ float red_smem[WARPS_PER_BLOCK * 2];

    // A vector of Q or K elements for the current timestep.
    using Qk_vec_k = typename Qk_vec_k_<T, Dh_MAX>::Type;  // with kernel-used precision
    using Qk_vec_m = typename Qk_vec_m_<T, Dh_MAX>::Type;  // with memory-used precision

    // Use alignment for safely casting the shared buffers as Qk_vec_k.
    // Shared memory to store Q inputs.
    __shared__ __align__(sizeof(Qk_vec_k)) Tk q_smem[Dh_MAX];

    // This is one of the reasons we should have a separate kernel for cross attention
    __shared__ __align__(sizeof(Qk_vec_k)) Tk bias_smem[DO_CROSS_ATTENTION ? Dh_MAX : 1];

    // The number of elements per vector.
    constexpr int QK_VEC_SIZE = sizeof(Qk_vec_m) / sizeof(T);
    // Make sure the hidden size per head is a multiple of the vector size.
    static_assert(Dh_MAX % QK_VEC_SIZE == 0, "");
    // We will use block wide reduction if needed
    // static_assert(Dh_MAX / QK_VEC_SIZE <= WARP_SIZE, "");
    // The number of vectors per warp.
    constexpr int QK_VECS_PER_WARP = Dh_MAX / QK_VEC_SIZE;

    // The layout of the cache is [B, H, Dh/x, L, x] with x == 4/8/16 for FP32/FP16/FP8. Since each thread
    // owns x elements, we have to decompose the linear index into chunks of x values and the posi-
    // tion of the thread in that chunk.

    // The number of elements in a chunk of 16B (that's the x in the above formula).
    constexpr int QK_ELTS_IN_16B = 16 / sizeof(T);
    // The number of K vectors in 16B.
    constexpr int QK_VECS_IN_16B = 16 / sizeof(Qk_vec_m);

    // The batch/beam idx
    const int bi = blockIdx.y;
    if (!SPLIT_KV_CACHE && params.finished != nullptr && params.finished[bi] == true) {
        return;
    }
    // The beam idx
    const int beami = bi % params.beam_width;
    // The "beam-aware" batch idx
    const int bbi = bi / params.beam_width;
    // The head.
    const int hi = blockIdx.x;
    // Combine the batch and the head indices.
    const int bhi = bi * params.num_heads + hi;
    // Combine the "beam-aware" batch idx and the head indices.
    const int bbhi = bbi * params.beam_width * params.num_heads + hi;
    // The thread in the block.
    const int tidx = threadIdx.x;
    // Layer stride in the block of KV Cache.
    const int layer_stride = params.num_heads * params.hidden_size_per_head * params.tokens_per_block;
    // Head stride in the block of KV Cache.
    const int head_stride      = params.hidden_size_per_head * params.tokens_per_block;
    const int tokens_per_block = params.tokens_per_block;
    const int layer_index      = params.layer_index;

    // timestep of current batch
    const int cur_timestep = get_cur_timestep<STEP_T, SPLIT_KV_CACHE>(params.timestep, bbi);

    const bool handle_kv = !DO_CROSS_ATTENTION || (DO_CROSS_ATTENTION && cur_timestep == 0);

    // While doing the product Q*K^T for the different keys we track the max.
    float qk_max = -FLT_MAX;

    float qk = 0.0F;

    int qkv_base_offset = (params.stride == 0) ? bhi * Dh : bi * params.stride + hi * Dh;

    const size_t bi_seq_len_offset = bi * params.memory_max_len;

    int tlength = (DO_CROSS_ATTENTION) ?
                      params.memory_length_per_sample[bi] - 1 :
                      (SPLIT_KV_CACHE) ? cur_timestep :
                                         (params.length_per_sample == nullptr) ?
                                         cur_timestep :
                                         params.length_per_sample[bi] + params.max_prefix_prompt_length;
    const int first_step   = max(0, tlength + 1 - params.memory_max_len);
    const int tlength_circ = tlength % params.memory_max_len;

    // First QK_VECS_PER_WARP load Q and K + the bias values for the current timestep.
    const bool is_masked = tidx >= QK_VECS_PER_WARP;

    // The offset in the Q and K buffer also accounts for the batch.
    int qk_offset = qkv_base_offset + tidx * QK_VEC_SIZE;
    // The offset in the bias buffer.
    int qk_bias_offset = hi * Dh + tidx * QK_VEC_SIZE;

    const bool do_ia3      = handle_kv && params.ia3_tasks != nullptr;
    const int  ia3_task_id = do_ia3 ? params.ia3_tasks[bbi] : 0;

#if ENABLE_INT8
    using T_scale = typename mmha::kv_cache_scale_type_t<T, Tcache>::Type;
    __shared__ float k_s_max, k_scale_f;
    __shared__ float v_s_max, v_scale_f;
    float            k_local_max = -FLT_MAX;
    float            v_local_max = -FLT_MAX;
#endif

    // Trigger the loads from the Q and K buffers.
    Qk_vec_k q;
    zero(q);
    if (!is_masked && (Dh == Dh_MAX || tidx * QK_VEC_SIZE < Dh)) {
        if (params.int8_mode == 2 || params.int8_mode == 3) {
#if ENABLE_INT8
            using Packed_Int8_t  = typename packed_type<int8_t, num_elems<Qk_vec_m>::value>::type;
            using Packed_Float_t = typename packed_type<float, num_elems<Qk_vec_m>::value>::type;
            const auto q_scaling = params.qkv_scale_out[0];
            const auto q_quant =
                *reinterpret_cast<const Packed_Int8_t*>(&reinterpret_cast<const int8_t*>(params.q)[qk_offset]);

            convert_from_float(q, mul<Packed_Float_t, float>(q_scaling, float_from_int8(q_quant)));
#endif
        }
        else {
            // q = vec_conversion<Qk_vec_k, Qk_vec_m>(*reinterpret_cast<const Qk_vec_m*>(&params.q[qk_offset]));
            q = vec_conversion<Qk_vec_k, Qk_vec_m>(ldg(reinterpret_cast<const Qk_vec_m*>(&params.q[qk_offset])));
        }
    }

    Qk_vec_k k;
    zero(k);
    if (DO_CROSS_ATTENTION) {
        // The 16B chunk written by the thread.
        int co = tidx / QK_VECS_IN_16B;
        // The position of the thread in that 16B chunk.
        int ci = tidx % QK_VECS_IN_16B * QK_VEC_SIZE;

        // Two chunks are separated by L * x elements. A thread write QK_VEC_SIZE elements.
        int head_offset = co * tokens_per_block * QK_ELTS_IN_16B + tlength % tokens_per_block * QK_ELTS_IN_16B + ci;
        // get (layer_index, hi, co, tlength, ci) of k_cache
        Tcache* cur_k_cache_ptr = get_kv_cache_ptr<Tcache, KV_CACHE_T, SPLIT_KV_CACHE>(
            inp_kcache, bi, layer_index, hi, tlength, tokens_per_block, layer_stride, head_stride, head_offset);
        // k = !is_masked && (Dh == Dh_MAX || tidx * QK_VEC_SIZE < Dh) ?
        //         vec_conversion<Qk_vec_k, Qk_vec_m>(*reinterpret_cast<const Qk_vec_m*>(cur_k_cache_ptr)) :
        //         k;
        k = !is_masked && (Dh == Dh_MAX || tidx * QK_VEC_SIZE < Dh) ?
                vec_conversion<Qk_vec_k, Qk_vec_m>(ldg(reinterpret_cast<const Qk_vec_m*>(cur_k_cache_ptr))) :
                k;
    }
    else {
        if (params.int8_mode == 2 || params.int8_mode == 3) {
            using Packed_Int8_t  = typename packed_type<int8_t, num_elems<Qk_vec_m>::value>::type;
            using Packed_Float_t = typename packed_type<float, num_elems<Qk_vec_m>::value>::type;
            const auto k_scaling = params.qkv_scale_out[1];
            const auto k_quant =
                *reinterpret_cast<const Packed_Int8_t*>(&reinterpret_cast<const int8_t*>(params.k)[qk_offset]);

            convert_from_float(k, mul<Packed_Float_t, float>(k_scaling, float_from_int8(k_quant)));
        }
        else {
            // k = !is_masked && (Dh == Dh_MAX || tidx * QK_VEC_SIZE < Dh) ?
            //         vec_conversion<Qk_vec_k, Qk_vec_m>(*reinterpret_cast<const Qk_vec_m*>(&params.k[qk_offset])) :
            //         k;
            k = !is_masked && (Dh == Dh_MAX || tidx * QK_VEC_SIZE < Dh) ?
                    vec_conversion<Qk_vec_k, Qk_vec_m>(ldg(reinterpret_cast<const Qk_vec_m*>(&params.k[qk_offset]))) :
                    k;
        }
    }

    // Trigger the loads from the Q and K bias buffers.
    Qk_vec_k q_bias;
    zero(q_bias);
    // q_bias =
    //     (!is_masked && Dh == Dh_MAX || tidx * QK_VEC_SIZE < Dh) && params.q_bias != nullptr ?
    //         vec_conversion<Qk_vec_k, Qk_vec_m>(*reinterpret_cast<const Qk_vec_m*>(&params.q_bias[qk_bias_offset])) :
    //         q_bias;
    q_bias =
        (!is_masked && Dh == Dh_MAX || tidx * QK_VEC_SIZE < Dh) && params.q_bias != nullptr ?
            vec_conversion<Qk_vec_k, Qk_vec_m>(ldg(reinterpret_cast<const Qk_vec_m*>(&params.q_bias[qk_bias_offset]))) :
            q_bias;

    Qk_vec_k k_bias;
    zero(k_bias);
    if (handle_kv) {
        k_bias =
            !is_masked && (Dh == Dh_MAX || tidx * QK_VEC_SIZE < Dh) && params.k_bias != nullptr ?
                vec_conversion<Qk_vec_k, Qk_vec_m>(ldg(reinterpret_cast<const Qk_vec_m*>(&params.k_bias[qk_bias_offset]))) :
                k_bias;
    }

    // Computes the Q/K values with bias.
    q = add(q, q_bias);
    if (handle_kv) {
        k = add(k, k_bias);
    }
    if (do_ia3 && !is_masked) {
        k = mul<Qk_vec_k, Qk_vec_k, Qk_vec_k>(
            k,
            vec_conversion<Qk_vec_k, Qk_vec_m>(ldg(reinterpret_cast<const Qk_vec_m*>(
                &params.ia3_key_weights[(ia3_task_id * params.num_heads + hi) * Dh + tidx * QK_VEC_SIZE]))));
    }

    // Padded len
    const int padd_len =
        (SPLIT_KV_CACHE) ? 0 : (params.total_padding_tokens == nullptr) ? 0 : params.total_padding_tokens[bi];

    if (params.rotary_embedding_dim > 0 && !params.neox_rotary_style) {
        if (handle_kv) {
            apply_rotary_embedding(q, k, tidx, params.rotary_embedding_dim, cur_timestep - padd_len);
        }
        else {
            apply_rotary_embedding(q, tidx, params.rotary_embedding_dim, cur_timestep - padd_len);
        }
    }
    else if (params.rotary_embedding_dim > 0 && params.neox_rotary_style) {
        const bool do_rotary = !is_masked && QK_VEC_SIZE * tidx < params.rotary_embedding_dim;

        T* q_smem = reinterpret_cast<T*>(smem_);
        T* k_smem = q_smem + params.rotary_embedding_dim;

        const int half_rotary_dim = params.rotary_embedding_dim / 2;
        const int half_idx        = (tidx * QK_VEC_SIZE) / half_rotary_dim;
        const int intra_half_idx  = (tidx * QK_VEC_SIZE) % half_rotary_dim;
        const int smem_pitch      = half_rotary_dim;  // TODO: adjust for bank conflicts

        assert(half_rotary_dim % QK_VEC_SIZE == 0);

        if (do_rotary) {
            *reinterpret_cast<Qk_vec_k*>(q_smem + half_idx * smem_pitch + intra_half_idx) = q;

            if (handle_kv) {
                *reinterpret_cast<Qk_vec_k*>(k_smem + half_idx * smem_pitch + intra_half_idx) = k;
            }
        }

        __syncthreads();

        const int     transpose_idx = half_idx * (half_rotary_dim / 2) + intra_half_idx / 2;
        constexpr int tidx_factor   = (QK_VEC_SIZE > 1) ? QK_VEC_SIZE / 2 : 1;
        if (do_rotary) {
            mmha::vec_from_smem_transpose(q, q_smem, transpose_idx, smem_pitch);

            if (handle_kv) {
                mmha::vec_from_smem_transpose(k, k_smem, transpose_idx, smem_pitch);

                mmha::apply_rotary_embedding(
                    q, k, transpose_idx / tidx_factor, params.rotary_embedding_dim, cur_timestep - padd_len);

                mmha::write_smem_transpose(k, k_smem, transpose_idx, smem_pitch);
            }
            else {
                mmha::apply_rotary_embedding(q, transpose_idx / tidx_factor, params.rotary_embedding_dim, cur_timestep);
            }
            mmha::write_smem_transpose(q, q_smem, transpose_idx, smem_pitch);
        }

        __syncthreads();

        if (do_rotary) {
            q = *reinterpret_cast<Qk_vec_k*>(q_smem + half_idx * smem_pitch + intra_half_idx);
            if (handle_kv) {
                k = *reinterpret_cast<Qk_vec_k*>(k_smem + half_idx * smem_pitch + intra_half_idx);
            }
        }

        __syncthreads();
    }

    if (handle_kv && ENABLE_8BITS_CACHE) {
#if ENABLE_INT8
        float* k_cache_scale = params.k_scale_cache_ptr[bi] + hi * params.memory_max_len;
        k_local_max          = mmha::fabs_max(k);
        k_local_max          = blockDim.x <= 32 ? warpReduceMax(k_local_max) : blockReduceMax(k_local_max);

        if (threadIdx.x == 0) {
            k_s_max                     = k_local_max;
            k_scale_f                   = 127 / k_s_max;
            k_cache_scale[tlength_circ] = 1.0 / k_scale_f;
        }
#endif
    }
    __syncthreads();

    if (!is_masked) {
        // Store the Q values to shared memory.
        *reinterpret_cast<Qk_vec_k*>(&q_smem[tidx * QK_VEC_SIZE]) = q;

        // Store Dh values of k_bias into smem, since will need to add later
        if (DO_CROSS_ATTENTION && cur_timestep == 0) {
            *reinterpret_cast<Qk_vec_k*>(&bias_smem[tidx * QK_VEC_SIZE]) = k_bias;
        }

        // Write the K values to the global memory cache.
        //
        // NOTE: The stores are uncoalesced as we have multiple chunks of 16B spread across the memory
        // system. We designed it this way as it allows much better memory loads (and there are many
        // more loads) + the stores are really "write and forget" since we won't need the ack before
        // the end of the kernel. There's plenty of time for the transactions to complete.

        // The 16B chunk written by the thread.
        int co = tidx / QK_VECS_IN_16B;
        // The position of the thread in that 16B chunk.
        int ci = tidx % QK_VECS_IN_16B * QK_VEC_SIZE;

        // Two chunks are separated by L * x elements. A thread write QK_VEC_SIZE elements.
        int head_offset =
            co * tokens_per_block * QK_ELTS_IN_16B + tlength_circ % tokens_per_block * QK_ELTS_IN_16B + ci;
        // get (layer_index, hi, co, tlength_circ, ci) of k_cache
        Tcache* cur_k_cache_ptr = get_kv_cache_ptr<Tcache, KV_CACHE_T, SPLIT_KV_CACHE>(
            inp_kcache, bi, layer_index, hi, tlength_circ, tokens_per_block, layer_stride, head_stride, head_offset);

        if (handle_kv) {
            // Trigger the stores to global memory.
            if (Dh == Dh_MAX || co < Dh / QK_ELTS_IN_16B) {
                if (!ENABLE_8BITS_CACHE) {
                    *reinterpret_cast<Qk_vec_m*>(cur_k_cache_ptr) = vec_conversion<Qk_vec_m, Qk_vec_k>(k);
                }
                else {
#if ENABLE_INT8
                    T_scale scaleOrigQuant;
                    mmha::convert_from_float(&scaleOrigQuant, k_scale_f);
                    // Store 8bits kv cache
                    mmha::store_8bits_kv_cache_vec(cur_k_cache_ptr, k, 0, scaleOrigQuant);
#endif
                }
            }
        }

        // Compute \sum_i Q[i] * K^T[i] for the current timestep.
#ifdef MMHA_USE_FP32_ACUM_FOR_FMA
        using Qk_vec_acum = typename Qk_vec_acum_fp32_<Qk_vec_k>::Type;
#else
        using Qk_vec_acum = typename Qk_vec_accum_<Qk_vec_k>::Type;
#endif
        qk = dot<Qk_vec_acum, Qk_vec_k>(q, k);
        if (QK_VECS_PER_WARP <= WARP_SIZE) {
#pragma unroll
            for (int mask = QK_VECS_PER_WARP / 2; mask >= 1; mask /= 2) {
                qk += __shfl_xor(qk, mask); // __shfl_xor_sync(shfl_mask(QK_VECS_PER_WARP), qk, mask);
            }
        }
    }

    if (QK_VECS_PER_WARP > WARP_SIZE) {
        constexpr int WARPS_PER_RED = (QK_VECS_PER_WARP + WARP_SIZE - 1) / WARP_SIZE;
        qk                          = block_sum<WARPS_PER_RED>(&red_smem[WARPS_PER_RED], qk);
    }

    // Store that value in shared memory. Keep the Q*K^T value in register for softmax.
    if (tidx == 0) {
        // Normalize qk.
        qk *= params.inv_sqrt_dh;
        if (params.relative_attention_bias != nullptr) {
            qk = add(qk,
                     params.relative_attention_bias[hi * params.relative_attention_bias_stride
                                                        * params.relative_attention_bias_stride
                                                    + (tlength - padd_len) * params.relative_attention_bias_stride
                                                    + (tlength - padd_len)]);
        }
        // We don't need to apply the linear position bias here since qi - ki = 0 yields the position bias 0.

        qk_max                        = qk;
        qk_smem[tlength - first_step] = qk;
    }

    // Make sure the data is in shared memory.
    __syncthreads();

    // The type of queries and keys for the math in the Q*K^T product.
    using K_vec_k = typename K_vec_k_<T, THREADS_PER_KEY>::Type;
    using K_vec_m = typename K_vec_m_<T, THREADS_PER_KEY>::Type;
    // The number of elements per vector.
    constexpr int K_VEC_SIZE = sizeof(K_vec_m) / sizeof(T);
    // Make sure the hidden size per head is a multiple of the vector size.
    static_assert(Dh_MAX % K_VEC_SIZE == 0, "");
    // The number of elements per thread.
    constexpr int K_ELTS_PER_THREAD = Dh_MAX / THREADS_PER_KEY;
    // The number of vectors per thread.
    constexpr int K_VECS_PER_THREAD = K_ELTS_PER_THREAD / K_VEC_SIZE;

    // The position the first key loaded by each thread from the cache buffer (for this B * H).
    int ko = tidx / THREADS_PER_KEY;
    // The position of the thread in the chunk of keys.
    int ki = tidx % THREADS_PER_KEY * K_VEC_SIZE;

    static_assert(Dh_MAX == THREADS_PER_KEY * K_VEC_SIZE * K_VECS_PER_THREAD);

    // Load the Q values from shared memory. The values are reused during the loop on K.
    K_vec_k q_vec[K_VECS_PER_THREAD];
#pragma unroll
    for (int ii = 0; ii < K_VECS_PER_THREAD; ++ii) {
        q_vec[ii] = *reinterpret_cast<const K_vec_k*>(&q_smem[ki + ii * THREADS_PER_KEY * K_VEC_SIZE]);
    }

    K_vec_k k_bias_vec[DO_CROSS_ATTENTION ? K_VECS_PER_THREAD : 1];
    if (DO_CROSS_ATTENTION && cur_timestep == 0) {
#pragma unroll
        for (int ii = 0; ii < K_VECS_PER_THREAD; ++ii) {
            k_bias_vec[ii] = *reinterpret_cast<const K_vec_k*>(&bias_smem[ki + ii * THREADS_PER_KEY * K_VEC_SIZE]);
        }
    }

    // The number of timesteps loaded per iteration.
    constexpr int K_PER_ITER = THREADS_PER_BLOCK / THREADS_PER_KEY;
    // The number of keys per warp.
    constexpr int K_PER_WARP = WARP_SIZE / THREADS_PER_KEY;

    // Pick a number of keys to make sure all the threads of a warp enter (due to shfl_sync).
    int ti_end = div_up(tlength - first_step, K_PER_WARP) * K_PER_WARP + first_step;

    // prefix prompt length if has
    const int prefix_prompt_length = (params.prefix_prompt_lengths == nullptr) ? 0 : params.prefix_prompt_lengths[bi];

    // Iterate over the keys/timesteps to compute the various (Q*K^T)_{ti} values.
    const int* beam_indices = HAS_BEAMS ? &params.cache_indir[bi_seq_len_offset] : nullptr;

    for (int ti = first_step + ko; ti < ti_end; ti += K_PER_ITER) {
        const int ti_circ = ti % params.memory_max_len;
        Tcache*   k_cache = get_kv_cache_ptr<Tcache, KV_CACHE_T, SPLIT_KV_CACHE>(
            inp_kcache, bi, layer_index, hi, ti_circ, tokens_per_block, layer_stride, head_stride, ki);
        bool is_mask = (SPLIT_KV_CACHE) ?
                           false :
                           (params.masked_tokens != nullptr) && params.masked_tokens[bi_seq_len_offset + ti];

        // The keys loaded from the key cache.
        K_vec_k k[K_VECS_PER_THREAD];
        K_vec_k k_vec_zero;
        zero(k_vec_zero);
#pragma unroll
        for (int ii = 0; ii < K_VECS_PER_THREAD; ++ii) {
            int        jj            = ii * params.memory_max_len + ti_circ;
            const bool within_bounds = (Dh == Dh_MAX || jj * QK_ELTS_IN_16B < Dh * params.memory_max_len);
            if (ti < tlength) {
                if (!within_bounds) {
                    k[ii] = k_vec_zero;
                }
                else {
                    if (HAS_BEAMS) {
                        assert(false);  // TODO: support beam size > 1
                        const int beam_offset = beam_indices[ti_circ] * params.num_heads * params.memory_max_len * Dh;
                        k[ii]                 = vec_conversion<K_vec_k, K_vec_m>(
                            (ldg(reinterpret_cast<const K_vec_m*>(&k_cache[beam_offset + jj * QK_ELTS_IN_16B]))));
                    }
                    else {
                        if (!ENABLE_8BITS_CACHE) {
                            // get (layer_index, hi, ii, ti_circ, ki) of k_cache
                            k[ii] = vec_conversion<K_vec_k, K_vec_m>((ldg(reinterpret_cast<const K_vec_m*>(
                                &k_cache[ii * tokens_per_block * QK_ELTS_IN_16B
                                         + ti_circ % params.tokens_per_block * QK_ELTS_IN_16B]))));
                        }
                        else {
#if ENABLE_INT8
                            float*  k_scale_ptr = params.k_scale_cache_ptr[bi] + hi * params.memory_max_len;
                            float   k_scale_f   = k_scale_ptr[ti_circ];
                            T_scale k_scale_quant_orig;
                            mmha::convert_from_float(&k_scale_quant_orig, k_scale_f);
                            mmha::load_8bits_kv_cache_vec(&k[ii],
                                                          k_cache,
                                                          ii * tokens_per_block * QK_ELTS_IN_16B
                                                              + ti_circ % params.tokens_per_block * QK_ELTS_IN_16B,
                                                          k_scale_quant_orig);
#endif
                        }
                    }
                }
                // add bias and update k_cache
                if (DO_CROSS_ATTENTION && cur_timestep == 0) {
                    assert(false);  // TODO: support cross attention
                    k[ii] = add(k[ii], k_bias_vec[ii]);

                    if (do_ia3) {
                        k[ii] = mul<K_vec_k, K_vec_k, K_vec_k>(
                            k[ii],
                            vec_conversion<K_vec_k, K_vec_m>(ldg(reinterpret_cast<const K_vec_m*>(
                                &params.ia3_key_weights[(ia3_task_id * params.num_heads + hi) * Dh + ki
                                                        + ii * THREADS_PER_KEY * K_VEC_SIZE]))));
                    }

                    if (Dh == Dh_MAX || jj * QK_ELTS_IN_16B < Dh * params.memory_max_len) {
                        *reinterpret_cast<K_vec_m*>(&k_cache[jj * QK_ELTS_IN_16B]) =
                            vec_conversion<K_vec_m, K_vec_k>(k[ii]);
                    }
                }
            }
        }

        // Perform the dot product and normalize qk.
        //
        // WARNING: ALL THE THREADS OF A WARP MUST ENTER!!!
        // asm volatile ("s_waitcnt vmcnt(0)");
        float qk = Qk_dot<T, THREADS_PER_KEY>::dot(q_vec, k) * params.inv_sqrt_dh;

        // Store the product to shared memory. There's one qk value per timestep. Update the max.
        if (ti < tlength && tidx % THREADS_PER_KEY == 0) {
            if (params.relative_attention_bias != nullptr) {
                qk = add(qk,
                         params.relative_attention_bias[hi * params.relative_attention_bias_stride
                                                            * params.relative_attention_bias_stride
                                                        + tlength * params.relative_attention_bias_stride + ti]);
            }
            if (params.linear_bias_slopes != nullptr) {
                // Apply the linear position bias: (ki - qi) * slope[hi].
                // The padding token locates between the input context and the generated tokens.
                // We need to remove the number of padding tokens in the distance computation.
                //   ti   : 0 1 2 3 4 5 6 7 8 9(tlength)
                //   token: i i i i p p p o o o where i=input, p=pad, o=output.
                // e.g. ti = 2, dist = (9 - 3) - 2 = 4.
                int   max_context_length = params.max_prefix_prompt_length + params.max_input_length;
                float dist               = (ti < max_context_length ? ti + padd_len : ti) - tlength;

                qk += mul<float, T, float>(params.linear_bias_slopes[hi], dist);
            }
            qk_max                   = is_mask ? qk_max : fmaxf(qk_max, qk);
            qk_smem[ti - first_step] = qk;
        }
    }

// Perform the final reduction to compute the max inside each warp.
//
// NOTE: In a group of THREADS_PER_KEY threads, the leader already has the max value for the
// group so it's not needed to run the reduction inside the group (again).
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask >= THREADS_PER_KEY; mask /= 2) {
        qk_max = fmaxf(qk_max, __shfl_xor( qk_max, mask));
    }

    // Decompose the thread index into warp and lane.
    const int warp = tidx / WARP_SIZE;
    const int lane = tidx % WARP_SIZE;

    // The warp leader writes the max to shared memory.
    if (lane == 0) {
        red_smem[warp] = qk_max;
    }

    // Make sure the products are in shared memory.
    __syncthreads();

    // The warps finalize the reduction.
    qk_max = lane < WARPS_PER_BLOCK ? red_smem[lane] : -FLT_MAX;
#pragma unroll
    for (int mask = WARPS_PER_BLOCK / 2; mask >= 1; mask /= 2) {
        qk_max = fmaxf(qk_max, __shfl_xor(qk_max, mask));
    }

    // Broadcast to all the threads in the warp.
    qk_max = __shfl(qk_max, 0);

    // Compute the logits and start the sum.
    float sum = 0.f;
    for (int ti = first_step + tidx; ti <= tlength; ti += THREADS_PER_BLOCK) {
        bool is_mask = (SPLIT_KV_CACHE) ?
                           false :
                           (params.masked_tokens != nullptr) && params.masked_tokens[bi_seq_len_offset + ti];
#ifdef FP8_MHA
        float logit = 0.f;
        if (FP8_MHA_KERNEL) {
            logit = is_mask ? 0.f :
                              __expf((qk_smem[ti - first_step] - qk_max) * params.query_weight_output_scale[0]
                                     * params.query_weight_output_scale[0]);
        }
        else {
            logit = is_mask ? 0.f : __expf(qk_smem[ti - first_step] - qk_max);
        }
#else
        float logit       = is_mask ? 0.f : __expf(qk_smem[ti - first_step] - qk_max);
#endif
        sum += logit;
        qk_smem[ti - first_step] = logit;
    }

    // Compute the sum.
    sum = block_sum<WARPS_PER_BLOCK>(&red_smem[WARPS_PER_BLOCK], sum);

    // Normalize the logits.
    float        inv_sum = __fdividef(1.f, sum + 1.e-6f);
    const size_t cross_attention_out_offset =
        params.is_return_cross_attentions ?
            bhi * params.max_decoder_seq_len * params.memory_max_len + params.max_timestep * params.memory_max_len :
            0;
    for (int ti = first_step + tidx; ti <= tlength; ti += THREADS_PER_BLOCK) {
        float logit = qk_smem[ti - first_step] * inv_sum;
        if (params.is_return_cross_attentions) {
            params.cross_attention_out[cross_attention_out_offset + ti] = logit;
        }
        convert_from_float(logits_smem[ti - first_step], logit);
    }

    // Put Values part below so we leverage __syncthreads
    // from the previous step

    // The number of elements per vector.
    constexpr int V_VEC_SIZE = Dh_MAX / THREADS_PER_VALUE;
    // A vector of V elements for the current timestep.
    using V_vec_k = typename V_vec_k_<T, V_VEC_SIZE>::Type;
    using V_vec_m = typename V_vec_m_<T, V_VEC_SIZE>::Type;

    // The value computed by this thread.
    int vo = tidx / THREADS_PER_VALUE;
    // The hidden dimensions computed by this particular thread.
    int vi = tidx % THREADS_PER_VALUE * V_VEC_SIZE;

    // The number of values processed per iteration of the loop.
    constexpr int V_PER_ITER = THREADS_PER_BLOCK / THREADS_PER_VALUE;

    // One group of threads computes the product(s) for the current timestep.
    V_vec_k v_bias;
    zero(v_bias);
    if (Dh == Dh_MAX || vi < Dh) {
        if (handle_kv) {
            if (vo == tlength % V_PER_ITER) {
                // Trigger the loads from the V bias buffer.
                if (params.v_bias != nullptr) {
                    v_bias = vec_conversion<V_vec_k, V_vec_m>(
                        ldg(reinterpret_cast<const V_vec_m*>(&params.v_bias[hi * Dh + vi])));
                }
                if (DO_CROSS_ATTENTION) {
                    *reinterpret_cast<V_vec_m*>(&bias_smem[vi]) = vec_conversion<V_vec_m, V_vec_k>(v_bias);
                }
            }
        }
    }

    // From previous, before values, step
    // Also make sure the logits are in shared memory.
    __syncthreads();

    // Values continued
#ifdef MMHA_USE_FP32_ACUM_FOR_OUT
    using V_vec_acum = typename V_vec_acum_fp32_<V_vec_k>::Type;
#else
    using V_vec_acum = V_vec_k;
#endif
    // The partial outputs computed by each thread.
    V_vec_acum out;
    zero(out);

    // Loop over the timesteps to compute the partial outputs.
    if (Dh == Dh_MAX || vi < Dh) {

        // Separate the ti < memory_max_len and ti > memory_max_len
        // to prevent ti % memory_len when ti < memory_len, and
        // the compiler cannot optimize the codes automatically.
        const int min_length = min(tlength, params.memory_max_len);
        for (int ti = first_step + vo; ti < min_length; ti += V_PER_ITER) {
            // Fetch offset based on cache_indir when beam sampling
            Tcache* v_cache = get_kv_cache_ptr<Tcache, KV_CACHE_T, SPLIT_KV_CACHE>(
                inp_vcache, bi, layer_index, hi, ti, tokens_per_block, layer_stride, head_stride, vi);
            const int beam_src    = HAS_BEAMS ? params.cache_indir[bi_seq_len_offset + ti] : 0;
            const int beam_offset = HAS_BEAMS ? beam_src * params.num_heads * params.memory_max_len * Dh : 0;
            // get (layer_index, hi, ti, vi) of v_cache.
            V_vec_k v;
            if (!ENABLE_8BITS_CACHE) {
                v = vec_conversion<V_vec_k, V_vec_m>(
                    ldg(reinterpret_cast<const V_vec_m*>(&v_cache[beam_offset + ti % tokens_per_block * Dh])));
            }
            else {
#if ENABLE_INT8
                float*  v_scale_ptr = params.v_scale_cache_ptr[bi] + hi * params.memory_max_len;
                float   v_scale_f   = v_scale_ptr[ti];
                T_scale v_scale_quant_orig;
                mmha::convert_from_float(&v_scale_quant_orig, v_scale_f);
                mmha::load_8bits_kv_cache_vec(
                    &v, v_cache, beam_offset + ti % tokens_per_block * Dh, v_scale_quant_orig);
#endif
            }

            // if (DO_CROSS_ATTENTION && cur_timestep == 0) {
                // assert(false);  // TODO: support cross attention
                // v = add(v, vec_conversion<V_vec_k, V_vec_m>(*reinterpret_cast<V_vec_m*>(&bias_smem[vi])));
                // if (do_ia3) {
                //     v = mul<V_vec_k, V_vec_k, V_vec_k>(
                //         v,
                //         ldg(reinterpret_cast<const V_vec_k*>(
                //             &params.ia3_value_weights[(ia3_task_id * params.num_heads + hi) * Dh + vi])));
                // }
                // *reinterpret_cast<V_vec_m*>(&v_cache[ti * Dh]) = vec_conversion<V_vec_m, V_vec_k>(v);
            // }
            // Load the logits from shared memory.
#if defined(MMHA_USE_FP32_ACUM_FOR_LOGITS)
            float logit = logits_smem[ti - first_step];
            out         = fma(logit, cast_to_float(v), out);
#else  // MMHA_USE_FP32_ACUM_FOR_LOGITS
#ifdef FP8_MHA
            Tk logit;
            if (FP8_MHA_KERNEL) {
                // NOTE: fake quantization
                // logit = vec_conversion<Tk, Tquant>(vec_conversion<Tquant, Tk>(mul<Tk, float, Tk>(1.0f /
                // params.attention_qk_scale[0], logits_smem[ti])));
                logit = logits_smem[ti - first_step];
            }
            else {
                logit = logits_smem[ti - first_step];
            }
            out = fma(logit, v, out);
#else   // FP8_MHA
            Tk logit = logits_smem[ti - first_step];
            out      = fma(logit, v, out);
#endif  // FP8_MHA
#endif  // MMHA_USE_FP32_ACUM_FOR_LOGITS
        }
        for (int ti = first_step + vo; ti < tlength; ti += V_PER_ITER) {
            if (ti < params.memory_max_len) {
                // handled by previous loop
                continue;
            }
            const int ti_circ = ti % params.memory_max_len;
            Tcache*   v_cache = get_kv_cache_ptr<Tcache, KV_CACHE_T, SPLIT_KV_CACHE>(
                inp_vcache, bi, layer_index, hi, ti_circ, tokens_per_block, layer_stride, head_stride, vi);

            // Fetch offset based on cache_indir when beam sampling
            const int beam_src    = HAS_BEAMS ? params.cache_indir[bi_seq_len_offset + ti_circ] : 0;
            const int beam_offset = HAS_BEAMS ? beam_src * params.num_heads * params.memory_max_len * Dh : 0;
            // get (layer_index, hi, ti_circ, vi) of v_cache.
            V_vec_k v;
            if (!ENABLE_8BITS_CACHE) {
                v = vec_conversion<V_vec_k, V_vec_m>(
                    ldg(reinterpret_cast<const V_vec_m*>(&v_cache[beam_offset + ti_circ % tokens_per_block * Dh])));
            }
            else {
#if ENABLE_INT8
                float*  v_scale_ptr = params.v_scale_cache_ptr[bi] + hi * params.memory_max_len;
                float   v_scale_f   = v_scale_ptr[ti_circ];
                T_scale v_scale_quant_orig;
                mmha::convert_from_float(&v_scale_quant_orig, v_scale_f);
                mmha::load_8bits_kv_cache_vec(
                    &v, v_cache, beam_offset + ti_circ % tokens_per_block * Dh, v_scale_quant_orig);
#endif
            }
            if (DO_CROSS_ATTENTION && cur_timestep == 0) {
                assert(false);  // TODO: support cross attention
                v = add(v, vec_conversion<V_vec_k, V_vec_m>(*reinterpret_cast<V_vec_m*>(&bias_smem[vi])));
                if (do_ia3) {
                    v = mul<V_vec_k, V_vec_k, V_vec_k>(
                        v,
                        ldg(reinterpret_cast<const V_vec_k*>(
                            &params.ia3_value_weights[(ia3_task_id * params.num_heads + hi) * Dh + vi])));
                }
                *reinterpret_cast<V_vec_m*>(&v_cache[ti * Dh]) = vec_conversion<V_vec_m, V_vec_k>(v);
            }
            // Load the logits from shared memory.
#if defined(MMHA_USE_FP32_ACUM_FOR_LOGITS)
            float logit = logits_smem[ti - first_step];
            out         = fma(logit, cast_to_float(v), out);
#else  // MMHA_USE_FP32_ACUM_FOR_LOGITS
#ifdef FP8_MHA
            Tk logit;
            if (FP8_MHA_KERNEL) {
                // NOTE: fake quantization
                // logit = vec_conversion<Tk, Tquant>(vec_conversion<Tquant, Tk>(mul<Tk, float, Tk>(1.0f /
                // params.attention_qk_scale[0], logits_smem[ti])));
                logit = logits_smem[ti - first_step];
            }
            else {
                logit = logits_smem[ti - first_step];
            }
            out = fma(logit, v, out);
#else   // FP8_MHA
            Tk logit = logits_smem[ti - first_step];
            out      = fma(logit, v, out);
#endif  // FP8_MHA
#endif  // MMHA_USE_FP32_ACUM_FOR_LOGITS
        }
    }

    V_vec_k v;
    zero(v);

    // One group of threads computes the product(s) for the current timestep.
    if (vo == tlength % V_PER_ITER && (Dh == Dh_MAX || vi < Dh)) {
        if (DO_CROSS_ATTENTION) {
            assert(false);  // TODO: support cross attention
            // v = vec_conversion<V_vec_k, V_vec_m>(*reinterpret_cast<const V_vec_m*>(&v_cache[tlength * Dh]));
        }
        else {
            // Trigger the loads from the V buffer.
            const auto v_offset = qkv_base_offset + vi;
            if (params.int8_mode == 2 || params.int8_mode == 3) {
                using Packed_Int8_t  = typename packed_type<int8_t, num_elems<V_vec_k>::value>::type;
                using Packed_Float_t = typename packed_type<float, num_elems<V_vec_k>::value>::type;
                const auto v_scaling = params.qkv_scale_out[2];
                const auto v_quant =
                    *reinterpret_cast<const Packed_Int8_t*>(&reinterpret_cast<const int8_t*>(params.v)[v_offset]);

                convert_from_float(v, mul<Packed_Float_t, float>(v_scaling, float_from_int8(v_quant)));
            }
            else {
                v = vec_conversion<V_vec_k, V_vec_m>(ldg(reinterpret_cast<const V_vec_m*>(&params.v[v_offset])));
            }
            // Trigger the loads from the V bias buffer.
            // V_vec v_bias = *reinterpret_cast<const V_vec*>(&params.v_bias[hi*Dh + vi]);
        }

        // Compute the V values with bias.
        if (handle_kv) {
            v = add(v, v_bias);

            if (do_ia3) {
                v = mul<V_vec_k, V_vec_k, V_vec_k>(
                    v,
                    ldg(reinterpret_cast<const V_vec_k*>(
                        &params.ia3_value_weights[(ia3_task_id * params.num_heads + hi) * Dh + vi])));
            }
        }
    }

    if (handle_kv && (Dh == Dh_MAX || vi < Dh) && ENABLE_8BITS_CACHE) {
#if ENABLE_INT7     
        v_local_max          = mmha::fabs_max(v);
        v_local_max          = blockDim.x <= 32 ? warpReduceMax(v_local_max) : blockReduceMax(v_local_max);
        float* v_cache_scale = params.v_scale_cache_ptr[bi] + hi * params.memory_max_len;
        if (threadIdx.x == 0) {
            v_s_max                     = v_local_max;
            v_scale_f                   = 127 / v_s_max;
            v_cache_scale[tlength_circ] = 1.0 / v_scale_f;
        }
#endif
    }
    __syncthreads();

    // Store the values with bias back to global memory in the cache for V.
    // get (layer_index, hi, tlength_circ, vi) of v_cache
    Tcache* v_cache = get_kv_cache_ptr<Tcache, KV_CACHE_T, SPLIT_KV_CACHE>(
        inp_vcache, bi, layer_index, hi, tlength_circ, tokens_per_block, layer_stride, head_stride, vi);
    if (handle_kv && (Dh == Dh_MAX || vi < Dh) && vo == tlength % V_PER_ITER) {
        if (!ENABLE_8BITS_CACHE) {
            *reinterpret_cast<V_vec_m*>(&v_cache[tlength_circ % tokens_per_block * Dh]) =
                vec_conversion<V_vec_m, V_vec_k>(v);
        }
        else if (ENABLE_8BITS_CACHE) {
#if ENABLE_INT8
            T_scale v_scaleOrigQuant;
            mmha::convert_from_float(&v_scaleOrigQuant, v_scale_f);
            // Store 8bits kv cache.
            mmha::store_8bits_kv_cache_vec(v_cache, v, tlength_circ % tokens_per_block * Dh, v_scaleOrigQuant);
#endif
        }
    }

    if (vo == tlength % V_PER_ITER && (Dh == Dh_MAX || vi < Dh)) {
        // Initialize the output value with the current timestep.
#if defined(MMHA_USE_FP32_ACUM_FOR_LOGITS)
        out = fma(logits_smem[tlength - first_step], cast_to_float(v), out);
#else  // MMHA_USE_FP32_ACUM_FOR_LOGITS
#ifdef FP8_MHA
        Tk logit;
        if (FP8_MHA_KERNEL) {
            // NOTE: fake quantization
            // logit = mul<Tk, float, Tk>(1.0f / params.attention_qk_scale[0], logits_smem[tlength]);
            logit = logits_smem[tlength - first_step];
        }
        else {
            logit = logits_smem[tlength - first_step];
        }
        out = fma(logit, v, out);
#else   // FP8_MHA
        out = fma(logits_smem[tlength - first_step], v, out);
#endif  // FP8_MHA
#endif  // MMHA_USE_FP32_ACUM_FOR_LOGITS
    }

    // Make sure we can start writing to shared memory.
    __syncthreads();

    // Run the final reduction amongst the different groups computing different partial outputs.
    if (Dh == Dh_MAX || vi < Dh) {
#pragma unroll
        for (int active_groups = V_PER_ITER; active_groups >= 2; active_groups /= 2) {

            // The midpoint in the number of active groups.
            int midpoint = active_groups / 2;

            // The upper part of active threads store to shared memory.
            if (vo >= midpoint && vo < active_groups && (Dh == Dh_MAX || vi < Dh)) {
#ifdef MMHA_USE_FP32_ACUM_FOR_OUT
                convert_from_float(*reinterpret_cast<V_vec_k*>(&out_smem[(vo - midpoint) * Dh + vi]), out);
#else
                *reinterpret_cast<V_vec_k*>(&out_smem[(vo - midpoint) * Dh + vi]) = out;
#endif
            }
            __syncthreads();
            // The bottom warps update their values.
            if (vo < midpoint && (Dh == Dh_MAX || vi < Dh)) {
                out = add(*reinterpret_cast<const V_vec_k*>(&out_smem[vo * Dh + vi]), out);
            }
            __syncthreads();
        }
    }

    // Output the final values.
    if (vo == 0 && (Dh == Dh_MAX || vi < Dh)) {
#ifdef MMHA_USE_FP32_ACUM_FOR_OUT
        if (FP8_MHA_KERNEL) {
#ifdef FP8_MHA
            // float result_scale = params.attention_qk_scale[0] * params.query_weight_output_scale[0] *
            // params.attention_output_weight_input_scale_inv[0];
            float result_scale =
                params.query_weight_output_scale[0] * params.attention_output_weight_input_scale_inv[0];
            convert_from_float(ldg(reinterpret_cast<V_vec_m*>(&params.out[bhi * Dh + vi]),
                               mul<V_vec_acum, float, V_vec_acum>(result_scale, out)));
#endif  // FP8_MHA
        }
        else if (params.int8_mode == 2 || params.int8_mode == 3) {
            using Packed_Int8_t = typename packed_type<int8_t, num_elems<V_vec_acum>::value>::type;
            out                 = mul<V_vec_acum, float>(*params.attention_out_scale, out);
            *reinterpret_cast<Packed_Int8_t*>(&(reinterpret_cast<int8_t*>(params.out)[bhi * Dh + vi])) =
                cast_to_int8(out);
        }
        else {
            convert_from_float(*reinterpret_cast<V_vec_m*>(&params.out[bhi * Dh + vi]), out);
        }
#else   // MMHA_USE_FP32_ACUM_FOR_OUT
        // TODO: support int8_mode?
        *reinterpret_cast<V_vec_m*>(&params.out[bhi * Dh + vi]) = vec_conversion<V_vec_m, V_vec_acum>(out);
#endif  // MMHA_USE_FP32_ACUM_FOR_OUT
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace mmha

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, int Dh, int Dh_MAX, typename KERNEL_PARAMS_TYPE, bool SPLIT_KV_CACHE = false>
void paged_mmha_launch_kernel(const KERNEL_PARAMS_TYPE& params, const hipStream_t& stream);

