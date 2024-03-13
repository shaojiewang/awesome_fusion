
#include "decoder_masked_multihead_attention.h"
#include "decoder_paged_masked_multihead_attention_template.hpp"
#include "decoder_paged_masked_multihead_attention_template_128.hpp"
#include "decoder_masked_multihead_attention_utils.h"
#include <assert.h>
#include <float.h>
#include <type_traits>

////////////////////////////////////////////////////////////////////////////////////////////////////

#define PAGED_MMHA_LAUNCH_KERNEL(T,                                                                                             \
                                 Tcache,                                                                                        \
                                 Dh,                                                                                            \
                                 Dh_MAX,                                                                                        \
                                 Dh_TILE_NUM,                                                                                   \
                                 THDS_PER_KEY,                                                                                  \
                                 THDS_PER_VALUE,                                                                                \
                                 THDS_PER_BLOCK,                                                                                \
                                 DO_CROSS_ATTENTION,                                                                            \
                                 HAS_BEAMS,                                                                                     \
                                 SPLIT_KV_CACHE,                                                                                \
                                 DO_MULTI_BLOCK,                                                                                \
                                 stream)                                                                                        \
    size_t seq_len_tile = mmha::multi_block_grid_setup<T, Dh, DO_CROSS_ATTENTION, SPLIT_KV_CACHE>(                              \
        params, THDS_PER_VALUE, THDS_PER_BLOCK, params.max_timestep, DO_MULTI_BLOCK);                                           \
    dim3 grid{static_cast<unsigned>(params.num_heads), static_cast<unsigned>(params.batch_size),                                \
        static_cast<unsigned>(seq_len_tile)};                                                                                   \
    size_t smem_sz =                                                                                                            \
        mmha::smem_size_in_bytes<T, DO_CROSS_ATTENTION, SPLIT_KV_CACHE, DO_MULTI_BLOCK>(params, THDS_PER_VALUE, THDS_PER_BLOCK);\
    if (smem_sz >= 64 * 1024)                                                                                                   \
    {                                                                                                                           \
        assert(false);                                                                                                          \
    }                                                                                                                           \
    mmha::paged_masked_multihead_attention_128_kernel<T,                                                                        \
                                                      Tcache,                                                                   \
                                                      Dh,                                                                       \
                                                      Dh_MAX,                                                                   \
                                                      Dh_TILE_NUM,                                                              \
                                                      THDS_PER_KEY,                                                             \
                                                      THDS_PER_VALUE,                                                           \
                                                      THDS_PER_BLOCK,                                                           \
                                                      DO_CROSS_ATTENTION,                                                       \
                                                      HAS_BEAMS,                                                                \
                                                      SPLIT_KV_CACHE,                                                           \
                                                      DO_MULTI_BLOCK><<<grid, THDS_PER_BLOCK, smem_sz, stream>>>(params)

////////////////////////////////////////////////////////////////////////////////////////////////////

// !!! Specialize the launcher for Cross attention
template<typename T, int Dh, int Dh_MAX, typename KERNEL_PARAMS_TYPE, bool SPLIT_KV_CACHE>
void paged_mmha_launch_kernel(const KERNEL_PARAMS_TYPE& params, const hipStream_t& stream)
{
    constexpr bool DO_CROSS_ATTENTION = std::is_same<KERNEL_PARAMS_TYPE, Cross_multihead_attention_params<T>>::value;
    int            tlength            = (DO_CROSS_ATTENTION) ? params.memory_max_len : params.max_timestep;

    const int kv_cache_quant_mode = params.kv_cache_quant_mode;
    const bool do_multi_block = params.enable_multi_block;
    printf("tlength, CROSS_ATTENTION = %d, %d\n", tlength, DO_CROSS_ATTENTION);
    if (kv_cache_quant_mode == 0) {
        if (!do_multi_block) {
            if (params.num_heads * params.batch_size <= 16) {
                constexpr int  Dh_TILE_NUM = 4;
                constexpr int  THREADS_PER_VALUE  = threads_per_value_t<T, Dh_MAX / Dh_TILE_NUM>::value;
                printf("%d\n", __LINE__);
                if (params.cache_indir == nullptr) {
                    if (tlength < 32) {
                        PAGED_MMHA_LAUNCH_KERNEL(T, T, Dh, Dh_MAX, Dh_TILE_NUM, 4, THREADS_PER_VALUE, 128, DO_CROSS_ATTENTION, false, SPLIT_KV_CACHE, false, stream);
                    }
                    else if (tlength < 512) {
                        PAGED_MMHA_LAUNCH_KERNEL(T, T, Dh, Dh_MAX, Dh_TILE_NUM, 2, THREADS_PER_VALUE, 256, DO_CROSS_ATTENTION, false, SPLIT_KV_CACHE, false, stream);
                            printf("seq_len_tile=%d, smem_sz=%d\n", seq_len_tile, smem_sz);
                    }
                    else {
                        if(params.batch_size * params.num_heads > 208) {
                            PAGED_MMHA_LAUNCH_KERNEL(T, T, Dh, Dh_MAX, Dh_TILE_NUM, 2, THREADS_PER_VALUE, 256, DO_CROSS_ATTENTION, false, SPLIT_KV_CACHE, false, stream);
                        } else {
                            PAGED_MMHA_LAUNCH_KERNEL(T, T, Dh, Dh_MAX, Dh_TILE_NUM, 2, THREADS_PER_VALUE, 1024, DO_CROSS_ATTENTION, false, SPLIT_KV_CACHE, false, stream);
                        }
                    }
                }
                else {
                    assert(false);
                }
            } else if (params.num_heads * params.batch_size <= 32) {
                constexpr int  Dh_TILE_NUM = 2;
                constexpr int  THREADS_PER_VALUE  = threads_per_value_t<T, Dh_MAX / Dh_TILE_NUM>::value;
                if (params.cache_indir == nullptr) {
                    if (tlength < 32) {
                        PAGED_MMHA_LAUNCH_KERNEL(T, T, Dh, Dh_MAX, Dh_TILE_NUM, 4, THREADS_PER_VALUE, 128, DO_CROSS_ATTENTION, false, SPLIT_KV_CACHE, false, stream);
                    }
                    else if (tlength < 512) {
                        PAGED_MMHA_LAUNCH_KERNEL(T, T, Dh, Dh_MAX, Dh_TILE_NUM, 2, THREADS_PER_VALUE, 256, DO_CROSS_ATTENTION, false, SPLIT_KV_CACHE, false, stream);
                    }
                    else {
                        if(params.batch_size * params.num_heads > 208) {
                            PAGED_MMHA_LAUNCH_KERNEL(T, T, Dh, Dh_MAX, Dh_TILE_NUM, 2, THREADS_PER_VALUE, 256, DO_CROSS_ATTENTION, false, SPLIT_KV_CACHE, false, stream);
                        } else {
                            PAGED_MMHA_LAUNCH_KERNEL(T, T, Dh, Dh_MAX, Dh_TILE_NUM, 2, THREADS_PER_VALUE, 1024, DO_CROSS_ATTENTION, false, SPLIT_KV_CACHE, false, stream);
                        }
                    }
                }
                else {
                    assert(false);
                }
            } else {
                constexpr int  Dh_TILE_NUM = 1;
                constexpr int  THREADS_PER_VALUE  = threads_per_value_t<T, Dh_MAX / Dh_TILE_NUM>::value;
                if (params.cache_indir == nullptr) {
                    if (tlength < 32) {
                        PAGED_MMHA_LAUNCH_KERNEL(T, T, Dh, Dh_MAX, Dh_TILE_NUM, 4, THREADS_PER_VALUE, 128, DO_CROSS_ATTENTION, false, SPLIT_KV_CACHE, false, stream);
                    }
                    else if (tlength < 512) {
                        PAGED_MMHA_LAUNCH_KERNEL(T, T, Dh, Dh_MAX, Dh_TILE_NUM, 2, THREADS_PER_VALUE, 256, DO_CROSS_ATTENTION, false, SPLIT_KV_CACHE, false, stream);
                    }
                    else {
                        if(params.batch_size * params.num_heads > 208) {
                            PAGED_MMHA_LAUNCH_KERNEL(T, T, Dh, Dh_MAX, Dh_TILE_NUM, 2, THREADS_PER_VALUE, 256, DO_CROSS_ATTENTION, false, SPLIT_KV_CACHE, false, stream);
                        } else {
                            PAGED_MMHA_LAUNCH_KERNEL(T, T, Dh, Dh_MAX, Dh_TILE_NUM, 2, THREADS_PER_VALUE, 1024, DO_CROSS_ATTENTION, false, SPLIT_KV_CACHE, false, stream);
                        }
                    }
                }
                else {
                    assert(false);
                }
            }
        } else {
            assert(false);
        }
    } else {
        assert(false);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#define INSTANTIATE_MMHA_LAUNCH_KERNEL(T, Dh, Dh_MAX, PARAMS_T, SPLIT_KV_CACHE)                                        \
    template void paged_mmha_launch_kernel<T, Dh, Dh_MAX, PARAMS_T<T>, SPLIT_KV_CACHE>(                      \
        const PARAMS_T<T>& params, const hipStream_t& stream)

INSTANTIATE_MMHA_LAUNCH_KERNEL(float, 128, 128, Paged_masked_multihead_attention_params, true);
// INSTANTIATE_MMHA_LAUNCH_KERNEL(float, 128, 128, Paged_masked_multihead_attention_params, false);
INSTANTIATE_MMHA_LAUNCH_KERNEL(uint16_t, 128, 128, Paged_masked_multihead_attention_params, true);
// INSTANTIATE_MMHA_LAUNCH_KERNEL(uint16_t, 128, 128, Paged_masked_multihead_attention_params, false);
INSTANTIATE_MMHA_LAUNCH_KERNEL(__nv_bfloat16, 128, 128, Paged_masked_multihead_attention_params, true);
// INSTANTIATE_MMHA_LAUNCH_KERNEL(__nv_bfloat16, 128, 128, Paged_masked_multihead_attention_params, false);

#undef INSTANTIATE_MMHA_LAUNCH_KERNEL
#undef PAGED_MMHA_LAUNCH_KERNEL

template<typename T, typename KERNEL_PARAMS_TYPE>
void paged_multihead_attention_(const KERNEL_PARAMS_TYPE& params, const hipStream_t& stream)
{
    switch (params.hidden_size_per_head) {
        case 128:
            paged_mmha_launch_kernel<T, 128, 128, KERNEL_PARAMS_TYPE, true>(params, stream);
            break;
        default:
            assert(false);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void paged_masked_multihead_attention(const Paged_masked_multihead_attention_params<float>& params, const hipStream_t& stream)
{
    paged_multihead_attention_<float, Paged_masked_multihead_attention_params<float>>(params, stream);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void paged_masked_multihead_attention(const Paged_masked_multihead_attention_params<uint16_t>& params, const hipStream_t& stream)
{
    paged_multihead_attention_<uint16_t, Paged_masked_multihead_attention_params<uint16_t>>(params, stream);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////

void paged_masked_multihead_attention(const Paged_masked_multihead_attention_params<__nv_bfloat16>& params, const hipStream_t& stream)
{
    paged_multihead_attention_<__nv_bfloat16, Paged_masked_multihead_attention_params<__nv_bfloat16>>(params, stream);
}

////////////////////////////////////////////////////////////////////////////////////////////////////


