#pragma once

#include "decoder_paged_masked_multihead_attention_template.hpp"

namespace mmha {
////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The type of the inputs. Supported types: float and half.
    typename T,
    // The type of the k/v cache.
    typename Tcache,
    // The hidden dimension per head.
    int Dh,
    int Dh_MAX,
    // Tile hidden dimension
    int Dh_TILE_NUM,
    // The number of threads per key.
    int THREADS_PER_KEY,
    // The number of threads per value.
    int THREADS_PER_VALUE,
    // The number of threads in a threadblock.
    int  THREADS_PER_BLOCK,
    bool DO_CROSS_ATTENTION,
    bool HAS_BEAMS,
    bool SPLIT_KV_CACHE = false,
    bool DO_MULTI_BLOCK = false>
__global__ void
paged_masked_multihead_attention_128_kernel(Paged_multihead_attention_params<T, DO_CROSS_ATTENTION, SPLIT_KV_CACHE> params)
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

#ifdef ENABLE_MULTI_BLOCK_OPTION
    constexpr bool MULTI_BLOCK_FLAG = DO_MULTI_BLOCK;
#else
    constexpr bool MULTI_BLOCK_FLAG = false;
#endif
    const auto max_time_step = static_cast<unsigned>(DO_MULTI_BLOCK ? params.timesteps_per_block : params.max_timestep);

    KV_CACHE_T* kv_blocks        = reinterpret_cast<KV_CACHE_T*>(params.kv_blocks);
    size_t**    kcache_bt_offset = params.k_cache;
    size_t**    vcache_bt_offset = params.v_cache;

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

    __shared__ float qk_current_smem[1];
    __shared__ Tk    logits_current_smem[1];

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
    constexpr int Dh_TILE_SIZE = Dh_MAX / Dh_TILE_NUM;
    const int hi = blockIdx.x / Dh_TILE_NUM;
    const int dhi = blockIdx.x % Dh_TILE_NUM;
    // const int hi = blockIdx.x;
    // Combine the batch and the head indices.
    const int bhi = bi * params.num_heads + hi;
    // Combine the "beam-aware" batch idx and the head indices.
    const int bbhi = bbi * params.beam_width * params.num_heads + hi;
    // The thread in the block.
    const int tidx = threadIdx.x;
    // Layer stride in the block of KV Cache.
    const int layer_stride = params.num_heads * params.hidden_size_per_head * params.tokens_per_block;
    // Head stride in the block of KV Cache.
    const int head_stride         = params.hidden_size_per_head * params.tokens_per_block;
    const int tokens_per_block    = params.tokens_per_block;
    const int layer_index         = params.layer_index;
    const int heads_per_gqa_group = params.heads_per_gqa_group;
    const int c_tile              = MULTI_BLOCK_FLAG ? blockIdx.z : 0;

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
    int       sample_tile  = MULTI_BLOCK_FLAG ? divUp(tlength, params.timesteps_per_block) : 1;

    if (MULTI_BLOCK_FLAG && c_tile >= sample_tile) {
        return;
    }
    const bool last_tile = (c_tile == sample_tile - 1) ? true : false;

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

    float* k_scale_ptr = nullptr;
    float* v_scale_ptr = nullptr;

    if (ENABLE_8BITS_CACHE) {
        k_scale_ptr = params.k_scale_cache_ptr[bi] + hi / heads_per_gqa_group * params.memory_max_len;
        v_scale_ptr = params.v_scale_cache_ptr[bi] + hi / heads_per_gqa_group * params.memory_max_len;
    }
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
        Tcache* cur_k_cache_ptr = get_kv_cache_ptr<Tcache>(kv_blocks,
                                                           kcache_bt_offset,
                                                           bi,
                                                           layer_index,
                                                           hi,
                                                           tlength,
                                                           tokens_per_block,
                                                           layer_stride,
                                                           head_stride,
                                                           head_offset,
                                                           heads_per_gqa_group);

        k = !is_masked && (Dh == Dh_MAX || tidx * QK_VEC_SIZE < Dh) ?
                vec_conversion<Qk_vec_k, Qk_vec_m>(*reinterpret_cast<const Qk_vec_m*>(cur_k_cache_ptr)) :
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
    q_bias = (!is_masked && Dh == Dh_MAX || tidx * QK_VEC_SIZE < Dh) && params.q_bias != nullptr ?
                 vec_conversion<Qk_vec_k, Qk_vec_m>(*reinterpret_cast<const Qk_vec_m*>(&params.q_bias[qk_bias_offset])) :
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
        Tcache* cur_k_cache_ptr = get_kv_cache_ptr<Tcache>(kv_blocks,
                                                           kcache_bt_offset,
                                                           bi,
                                                           layer_index,
                                                           hi,
                                                           tlength_circ,
                                                           tokens_per_block,
                                                           layer_stride,
                                                           head_stride,
                                                           head_offset,
                                                           heads_per_gqa_group);

        if (handle_kv && hi % heads_per_gqa_group == 0) {
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

    const auto timesteps_per_block = params.timesteps_per_block;


    // Pick a number of keys to make sure all the threads of a warp enter (due to shfl_sync).
    int ti_end = MULTI_BLOCK_FLAG ? div_up(timesteps_per_block, K_PER_WARP) * K_PER_WARP :
                                    div_up(tlength - first_step, K_PER_WARP) * K_PER_WARP
                                        + first_step;  // first_step seems 0 all the time

    // prefix prompt length if has
    const int prefix_prompt_length = (params.prefix_prompt_lengths == nullptr) ? 0 : params.prefix_prompt_lengths[bi];

    // Iterate over the keys/timesteps to compute the various (Q*K^T)_{ti} values.
    const int* beam_indices = HAS_BEAMS ? &params.cache_indir[bi_seq_len_offset] : nullptr;

    const auto c_tile_times_timesteps_per_block = c_tile * timesteps_per_block;
    const int  tile_offset                      = MULTI_BLOCK_FLAG ? c_tile_times_timesteps_per_block : 0;

    for (int ti = first_step + ko; ti < ti_end; ti += K_PER_ITER) {
        const int ti_circ = ti % params.memory_max_len;
        const int valid_ti_circ = min(ti_circ, tlength - 1);
        Tcache*   k_cache       = get_kv_cache_ptr<Tcache>(kv_blocks,
                                                   kcache_bt_offset,
                                                   bi,
                                                   layer_index,
                                                   hi,
                                                   valid_ti_circ,
                                                   tokens_per_block,
                                                   layer_stride,
                                                   head_stride,
                                                   ki,
                                                   heads_per_gqa_group);
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
                                         + valid_ti_circ % params.tokens_per_block * QK_ELTS_IN_16B]))));
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
        if (ti_circ < tlength && tidx % THREADS_PER_KEY == 0) {
            if (params.relative_attention_bias != nullptr) {
                qk = add(qk,
                         params.relative_attention_bias[hi * params.relative_attention_bias_stride
                                                            * params.relative_attention_bias_stride
                                                        + tlength * params.relative_attention_bias_stride + ti_circ]);
            }
            if (params.linear_bias_slopes != nullptr) {
                // Apply the linear position bias: (ki - qi) * slope[hi].
                // The padding token locates between the input context and the generated tokens.
                // We need to remove the number of padding tokens in the distance computation.
                //   ti   : 0 1 2 3 4 5 6 7 8 9(tlength)
                //   token: i i i i p p p o o o where i=input, p=pad, o=output.
                // e.g. ti = 2, dist = (9 - 3) - 2 = 4.
                int   max_context_length = params.max_prefix_prompt_length + params.max_input_length;
                float dist               = (ti_circ < max_context_length ? ti_circ + padd_len : ti_circ) - tlength;

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
    constexpr int V_VEC_SIZE = Dh_TILE_SIZE / THREADS_PER_VALUE;
    // A vector of V elements for the current timestep.
    using V_vec_k = typename V_vec_k_<T, V_VEC_SIZE>::Type;
    using V_vec_m = typename V_vec_m_<T, V_VEC_SIZE>::Type;

    // The value computed by this thread.
    int vo = tidx / THREADS_PER_VALUE;
    // The hidden dimensions computed by this particular thread.
    int vi = tidx % THREADS_PER_VALUE * V_VEC_SIZE + dhi * Dh_TILE_SIZE;

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
        int       context_v_loop_end = MULTI_BLOCK_FLAG ? timesteps_per_block : min_length;
        int       time_now           = 0;
        float     v_scale_f          = 1.0f;
        for (int ti = first_step + vo; ti < context_v_loop_end; ti += V_PER_ITER) {
            int time_now    = ti + tile_offset;
            time_now        = min(time_now, tlength - 1);
            Tcache* v_cache = get_kv_cache_ptr<Tcache>(kv_blocks,
                                                       vcache_bt_offset,
                                                       bi,
                                                       layer_index,
                                                       hi,
                                                       time_now,
                                                       tokens_per_block,
                                                       layer_stride,
                                                       head_stride,
                                                       vi,
                                                       heads_per_gqa_group);

            // Fetch offset based on cache_indir when beam sampling
            const int beam_src    = HAS_BEAMS ? params.cache_indir[bi_seq_len_offset + time_now] : 0;
            const int beam_offset = HAS_BEAMS ? beam_src * params.num_heads * params.memory_max_len * Dh : 0;
            // get (layer_index, hi, ti, vi) of v_cache.
            V_vec_k v;
            if (!ENABLE_8BITS_CACHE) {
                v = vec_conversion<V_vec_k, V_vec_m>(
                    ldg(reinterpret_cast<const V_vec_m*>(&v_cache[beam_offset + time_now % tokens_per_block * Dh])));
            }
            else {
#if ENABLE_INT8
                v_scale_f   = v_scale_ptr[time_now];
#endif
            }

            int        local_time_idx = ti;
            int        time_idx       = local_time_idx + (MULTI_BLOCK_FLAG ? c_tile_times_timesteps_per_block : 0);
            const bool is_mask = (MULTI_BLOCK_FLAG && local_time_idx >= timesteps_per_block) || (time_idx >= tlength);
            // Load the logits from shared memory.
            // Note that fma will convert 8bit vec to the accumulation data type (float by default).
            Logit_value_fma<Tk, V_vec_acum, V_vec_m, ENABLE_8BITS_CACHE, false>(
                out, reinterpret_cast<Tk*>(logits_smem + ti - first_step), v, v_scale_f, is_mask);
        }

        for (int ti = first_step + vo; ti < context_v_loop_end; ti += V_PER_ITER) {
            ti = MULTI_BLOCK_FLAG ? ti + c_tile_times_timesteps_per_block : ti; 
            if (ti < params.memory_max_len) {
                // handled by previous loop
                continue;
            }
            assert(false);
            const int ti_circ = ti % params.memory_max_len;
            Tcache*   v_cache = get_kv_cache_ptr<Tcache>(kv_blocks,
                                                       vcache_bt_offset,
                                                       bi,
                                                       layer_index,
                                                       hi,
                                                       ti_circ,
                                                       tokens_per_block,
                                                       layer_stride,
                                                       head_stride,
                                                       vi,
                                                       heads_per_gqa_group);

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
    if (vo == tlength % V_PER_ITER && (Dh == Dh_MAX || vi < Dh) && (!MULTI_BLOCK_FLAG || last_tile)) {
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

    if (handle_kv && (Dh == Dh_MAX || vi < Dh) && (!MULTI_BLOCK_FLAG || last_tile) && ENABLE_8BITS_CACHE) {
#if ENABLE_INT8
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
    Tcache* v_cache = get_kv_cache_ptr<Tcache>(kv_blocks,
                                               vcache_bt_offset,
                                               bi,
                                               layer_index,
                                               hi,
                                               tlength_circ,
                                               tokens_per_block,
                                               layer_stride,
                                               head_stride,
                                               vi,
                                               heads_per_gqa_group);
    if (handle_kv && (Dh == Dh_MAX || vi < Dh) && vo == tlength % V_PER_ITER && (!MULTI_BLOCK_FLAG || last_tile)) {
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

    if (vo == tlength % V_PER_ITER && (Dh == Dh_MAX || vi < Dh) && (!MULTI_BLOCK_FLAG || last_tile)) {
        // Initialize the output value with the current timestep.
#if defined(MMHA_USE_FP32_ACUM_FOR_LOGITS)
        if (!MULTI_BLOCK_FLAG) {
            out = fma(logits_smem[tlength], cast_to_float(v), out);
        }
        else {
            out = fma(logits_current_smem[0], cast_to_float(v), out);
        }
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
        if (!MULTI_BLOCK_FLAG) {
            out = fma(logits_smem[tlength - first_step], v, out);
        }
        else {
            out = fma(logits_current_smem[0], v, out);
        }
#endif  // FP8_MHA
#endif  // MMHA_USE_FP32_ACUM_FOR_LOGITS
    }

    // Make sure we can start writing to shared memory.
    __syncthreads();

    const auto bhi_seq_len_tile = bhi * params.max_seq_len_tile;

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
            if (!MULTI_BLOCK_FLAG) {
                V_vec_k final_out;
                convert_from_float(final_out, out);
                *reinterpret_cast<V_vec_k*>(&params.out[bhi * Dh + vi]) = final_out;
            }
            else {
                // for write partial output to partial_out
                int partial_out_offset = c_tile * params.batch_size * params.num_heads * params.hidden_size_per_head;
                // for write partial statistics to partial_max and partial_sum
                int partial_stats_offset = bhi_seq_len_tile + c_tile;

                // This makes sure we have coalesced memory access.
                V_vec_k partial_out;
                convert_from_float(partial_out, out);
                *reinterpret_cast<V_vec_k*>(&params.partial_out[partial_out_offset + bhi * Dh + vi]) = partial_out;
                convert_from_float(*reinterpret_cast<float*>(&params.partial_max[partial_stats_offset]), qk_max);
                convert_from_float(*reinterpret_cast<float*>(&params.partial_sum[partial_stats_offset]), sum);
            }
        }
#else   // MMHA_USE_FP32_ACUM_FOR_OUT
        // TODO: support int8_mode?
        *reinterpret_cast<V_vec_m*>(&params.out[bhi * Dh + vi]) = vec_conversion<V_vec_m, V_vec_acum>(out);
#endif  // MMHA_USE_FP32_ACUM_FOR_OUT
    }

#ifdef ENABLE_MULTI_BLOCK_OPTION
    if constexpr(MULTI_BLOCK_FLAG) {

        // hip::atomic_ref<int, cuda::thread_scope_device> count_ref{params.block_counter[bhi]};
        bool                                             last_block{false};
        if (tidx == 0) {
            if (__atomic_fetch_add(&(params.block_counter[bhi]), 1, __ATOMIC_RELAXED) == (sample_tile - 1)) {
                last_block = true;
            }
        }

        ////////////////////
        ////////////////////
        // Make sure every threadblock finishes the previous computation, and enter the last threadblock in the
        // following (for each B and H) Do the final computation in the last threadblock Final reduction computation
        // by combining all the partial max/sum and outputs
        ////////////////////
        ////////////////////
        if (__syncthreads_or(last_block)) {

            ////////////////////
            // Find the global max from all partial max -> use CUB BlockReduce
            ////////////////////

            float final_max          = -FLT_MAX;
            float thread_partial_max = -FLT_MAX;
            if (tidx < sample_tile)
                thread_partial_max = params.partial_max[bhi_seq_len_tile + tidx];
            // final_max = fmaxf(final_max, thread_partial_max);

            // Make sure we can start writing to shared memory.
            __syncthreads();

            // Specialize BlockReduce for a 1D block of THREADS_PER_BLOCK threads of type int
            //typedef cub::BlockReduce<float, THREADS_PER_BLOCK> BlockReduce;
            // Allocate shared memory for BlockReduce
            //__shared__ typename BlockReduce::TempStorage temp_storage;
            // Obtain a segment of consecutive items that are blocked across threads (final_max from above)
            // Compute the block-wide max for thread0
            //final_max = BlockReduce(temp_storage).Reduce(thread_partial_max, cub::Max(), sample_tile);
            
            final_max = blockReduceMax(thread_partial_max);

            __shared__ float final_max_smem;
            if (tidx == 0) {
                final_max_smem = final_max;
            }
            __syncthreads();

            // Finish the final_max computation
            final_max = final_max_smem;

            ////////////////////
            // Reduction for global sum over all partial sum (scaled by the exponential term from global max) -> use
            // gridDim.z threads
            ////////////////////

            float final_sum = 0.f;
            if (tidx < sample_tile) {
                thread_partial_max            = params.partial_max[bhi_seq_len_tile + tidx];
                const auto thread_partial_sum = params.partial_sum[bhi_seq_len_tile + tidx];
                final_sum += __expf(thread_partial_max - final_max) * thread_partial_sum;
            }

            // Compute the final_sum.
            final_sum = block_sum<WARPS_PER_BLOCK>(&red_smem[WARPS_PER_BLOCK], final_sum);

            ////////////////////
            // Reduction for final output (scaled by the exponential term from global max) -> use THREADS_PER_VALUE
            // * gridDim.z threads
            ////////////////////

            // Shared memory to store partial outputs for each oi. -> size: gridDim.z * Dh * 4 Bytes. Reuse qk_smem.
            T* out_oi_smem = reinterpret_cast<T*>(smem_);

            // Number of threads to utilize: THREADS_PER_VALUE * gridDim.z (THREADS_PER_VALUE for vectorized output
            // and gridDim.z for all the partial outputs)
            int threads_boundary = THREADS_PER_VALUE * sample_tile;  // should be smaller than THREADS_PER_BLOCK
            assert(threads_boundary <= THREADS_PER_BLOCK);

            const auto o_idx = chunk_index<T, V_vec_k, THREADS_PER_VALUE>(tidx);
            // The partial output region this thread takes care of
            const auto oo = o_idx.x;
            // The hidden dimensions computed by this particular thread. (refer to vi)
            const auto oi = o_idx.y;

            // Load partial output
            int thread_partial_out_offset = oo * params.batch_size * params.num_heads * params.hidden_size_per_head;
            // Load partial max (different to thread_partial_max since the threadIdx rule changes here)
            float thread_partial_max_for_out = params.partial_max[bhi_seq_len_tile + oo];

            // Load the partial outputs.
            V_vec_k thread_partial_out =
                *reinterpret_cast<const V_vec_k*>(&params.partial_out[thread_partial_out_offset + bhi * Dh + oi]);

            if (tidx >= threads_boundary) {
                zero(thread_partial_out);
                thread_partial_max_for_out = final_max;
            }

            Tk factor_compute;
            convert_from_float(factor_compute, __expf(thread_partial_max_for_out - final_max));

            thread_partial_out = mul<V_vec_k, Tk, V_vec_k>(factor_compute, thread_partial_out);

            // Make sure we can start writing to shared memory.
            __syncthreads();

            // The reduction iteration should start with a number which is a power of 2
            const auto reduction_iteration =
                static_cast<int>(math::next_power_of_two(static_cast<uint32_t>(sample_tile)));

            // Run the final reduction amongst the different groups computing different partial outputs.
// #pragma unroll
            for (int active_groups = reduction_iteration; active_groups >= 2; active_groups /= 2) {

                // The midpoint in the number of active groups.
                int midpoint = active_groups / 2;

                // The upper part of active threads store to shared memory.
                if (oo >= midpoint && oo < active_groups && (Dh == Dh_MAX || oi < Dh)) {
                    *reinterpret_cast<V_vec_k*>(&out_oi_smem[(oo - midpoint) * Dh + oi]) = thread_partial_out;
                }
                __syncthreads();

                // The bottom warps update their values.
                if (oo < midpoint && (Dh == Dh_MAX || oi < Dh)) {
                    thread_partial_out =
                        add(thread_partial_out, *reinterpret_cast<const V_vec_k*>(&out_oi_smem[oo * Dh + oi]));
                }
                __syncthreads();
            }

            ////////////////////
            // Final output O * inv_sum
            ////////////////////

            if (oo == 0 && (Dh == Dh_MAX || oi < Dh)) {
                const auto inv_sum = __fdividef(1.f, final_sum + 1.e-6f);
                Tk         inv_sum_compute;
                convert_from_float(inv_sum_compute, inv_sum);

                thread_partial_out = mul<V_vec_k, Tk, V_vec_k>(inv_sum_compute, thread_partial_out);

                *reinterpret_cast<V_vec_k*>(&params.out[bhi * Dh + oi]) = thread_partial_out;
            }

            // Reset qk_current_smem and block_counter for the next timestep
            if (tidx == 0) {
                params.block_counter[bhi] = 0;
            }
        }
    }
#endif  // ENABLE_MULTI_BLOCK_OPTION
}
////////////////////////////////////////////////////////////////////////////////////////////////////
}// namespace mmha

