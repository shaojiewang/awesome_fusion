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

#include "decoder_masked_multihead_attention.h"
#include "decoder_masked_multihead_attention_utils.h"
#include "hip_bf16_wrapper.h"
#include "hip_type_utils.cuh"
#include <hip/hip_fp16.h>
#include <stdint.h>

namespace mmha {

inline __device__ void print_vec_info(
    const int bid_x, const int bid_y, const int seq_id, const int tid, const float val, const float append = 0)
{
    printf("pvi block and thread index is: %d, %d, %d, %d, value: (%f), append value: %f \n",
           bid_x,
           bid_y,
           seq_id,
           tid,
           val,
           append);
}

inline __device__ void print_vec_info(
    const int bid_x, const int bid_y, const int seq_id, const int tid, const float2 val, const float append = 0)
{
    printf("pvi block and thread index is: %d, %d, %d, %d, value: (%f, %f), append value: %f \n",
           bid_x,
           bid_y,
           seq_id,
           tid,
           val.x,
           val.y,
           append);
}

inline __device__ void print_vec_info(
    const int bid_x, const int bid_y, const int seq_id, const int tid, const float4 val, const float append = 0)
{
    printf("pvi block and thread index is: %d, %d, %d, %d, value: (%f, %f, %f, %f), append value: %f \n",
           bid_x,
           bid_y,
           seq_id,
           tid,
           val.x,
           val.y,
           val.z,
           val.w,
           append);
}

inline __device__ void print_vec_info(
    const int bid_x, const int bid_y, const int seq_id, const int tid, const Float4_ val, const float append = 0)
{
    printf("pvi block and thread index is: %d, %d, %d, %d, value: (%f, %f, %f, %f), append value: %f \n",
           bid_x,
           bid_y,
           seq_id,
           tid,
           val.x.x,
           val.x.y,
           val.y.x,
           val.y.y,
           append);
}

inline __device__ void print_vec_info(
    const int bid_x, const int bid_y, const int seq_id, const int tid, const Float8_ val, const float append = 0)
{
    printf(
        "pvi block and thread index is: %d, %d, %d, %d, value: (%f, %f, %f, %f, %f, %f, %f, %f), append value: %f \n",
        bid_x,
        bid_y,
        seq_id,
        tid,
        val.x.x,
        val.x.y,
        val.y.x,
        val.y.y,
        val.z.x,
        val.z.y,
        val.w.x,
        val.w.y,
        append);
}

inline __device__ void print_vec_info(
    const int bid_x, const int bid_y, const int seq_id, const int tid, const uint32_t val, const float append = 0)
{
    float2 ele_x = half2_to_float2(val);
    printf("pvi block and thread index is: %d, %d, %d, %d, value: (%f, %f), append value: %f \n",
           bid_x,
           bid_y,
           seq_id,
           tid,
           ele_x.x,
           ele_x.y,
           append);
}

inline __device__ void print_vec_info(
    const int bid_x, const int bid_y, const int seq_id, const int tid, const uint2 val, const float append = 0)
{
    float2 ele_x = half2_to_float2(val.x);
    float2 ele_y = half2_to_float2(val.y);
    printf("pvi block and thread index is: %d, %d, %d, %d, value: (%f, %f, %f, %f), append value: %f \n",
           bid_x,
           bid_y,
           seq_id,
           tid,
           ele_x.x,
           ele_x.y,
           ele_y.x,
           ele_y.y,
           append);
}

inline __device__ void print_vec_info(
    const int bid_x, const int bid_y, const int seq_id, const int tid, const uint4 val, const float append = 0)
{
    float2 ele_x = half2_to_float2(val.x);
    float2 ele_y = half2_to_float2(val.y);
    float2 ele_z = half2_to_float2(val.z);
    float2 ele_w = half2_to_float2(val.w);
    printf(
        "pvi block and thread index is: %d, %d, %d, %d, value: (%f, %f, %f, %f, %f, %f, %f, %f), append value: %f \n",
        bid_x,
        bid_y,
        seq_id,
        tid,
        ele_x.x,
        ele_x.y,
        ele_y.x,
        ele_y.y,
        ele_z.x,
        ele_z.y,
        ele_w.x,
        ele_w.y,
        append);
}

inline __device__ void print_vec_info(
    const int bid_x, const int bid_y, const int seq_id, const int tid, const __nv_bfloat162 val, const float append = 0)
{
    float2 ele_f = bf1622float2(val);
    printf("pvi block and thread index is: %d, %d, %d, %d, value: (%f, %f), append value: %f \n",
           bid_x,
           bid_y,
           seq_id,
           tid,
           ele_f.x,
           ele_f.y,
           append);
}

inline __device__ void print_vec_info(
    const int bid_x, const int bid_y, const int seq_id, const int tid, const bf16_4_t val, const float append = 0)
{
    float2 ele_x = bf1622float2(val.x);
    float2 ele_y = bf1622float2(val.y);
    printf("pvi block and thread index is: %d, %d, %d, %d, value: (%f, %f, %f, %f), append value: %f \n",
           bid_x,
           bid_y,
           seq_id,
           tid,
           ele_x.x,
           ele_x.y,
           ele_y.x,
           ele_y.y,
           append);
}

inline __device__ void print_vec_info(
    const int bid_x, const int bid_y, const int seq_id, const int tid, const bf16_8_t val, const float append = 0)
{
    float2 ele_x = bf1622float2(val.x);
    float2 ele_y = bf1622float2(val.y);
    float2 ele_z = bf1622float2(val.z);
    float2 ele_w = bf1622float2(val.w);
    printf(
        "pvi block and thread index is: %d, %d, %d, %d, value: (%f, %f, %f, %f, %f, %f, %f, %f), append value: %f \n",
        bid_x,
        bid_y,
        seq_id,
        tid,
        ele_x.x,
        ele_x.y,
        ele_y.x,
        ele_y.y,
        ele_z.x,
        ele_z.y,
        ele_w.x,
        ele_w.y,
        append);
}

inline __device__ void convert_from_float(float* dst, float src)
{
    *dst = src;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_float(uint16_t* dst, float src)
{
    *dst = float_to_half(src);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_float(uint32_t* dst, float2 src)
{
    *dst = float2_to_half2(src);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef ENABLE_BF16
inline __device__ void convert_from_float(__nv_bfloat16* dst, float src)
{
    *dst = __float2bfloat16(src);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_float(__nv_bfloat162* dst, float2 src)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    *dst = __float22bfloat162_rn(src);
#else
    *dst   = __floats2bfloat162_rn(src.x, src.y);
#endif
}
#endif  // ENABLE_BF16
////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_float(uint2* dst, Float4_ src)
{
    dst->x = float2_to_half2(src.x);
    dst->y = float2_to_half2(src.y);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_float(uint2* dst, float4 src)
{
    convert_from_float(dst, Float4_{make_float2(src.x, src.y), make_float2(src.z, src.w)});
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_float(uint4* dst, Float8_ src)
{
    dst->x = float2_to_half2(src.x);
    dst->y = float2_to_half2(src.y);
    dst->z = float2_to_half2(src.z);
    dst->w = float2_to_half2(src.w);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef ENABLE_BF16
inline __device__ void convert_from_float(bf16_4_t* dst, Float4_ src)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    dst->x = __float22bfloat162_rn(src.x);
    dst->y = __float22bfloat162_rn(src.y);
#else
    dst->x = __floats2bfloat162_rn(src.x.x, src.x.y);
    dst->y = __floats2bfloat162_rn(src.y.x, src.y.y);
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_float(bf16_4_t* dst, float4 src)
{
    convert_from_float(dst, Float4_{make_float2(src.x, src.y), make_float2(src.z, src.w)});
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_float(bf16_8_t* dst, Float8_ src)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    dst->x = __float22bfloat162_rn(src.x);
    dst->y = __float22bfloat162_rn(src.y);
    dst->z = __float22bfloat162_rn(src.z);
    dst->w = __float22bfloat162_rn(src.w);
#else
    dst->x = __floats2bfloat162_rn(src.x.x, src.x.y);
    dst->y = __floats2bfloat162_rn(src.y.x, src.y.y);
    dst->z = __floats2bfloat162_rn(src.z.x, src.z.y);
    dst->w = __floats2bfloat162_rn(src.w.x, src.w.y);
#endif
}
#endif  // ENABLE_BF16

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef ENABLE_FP8
inline __device__ void convert_from_float(fp8_4_t* dst, float4 src)
{
    *dst = fp8_4_t(src);
}

inline __device__ void convert_from_float(fp8_2_t* dst, float2 src)
{
    *dst = fp8_2_t(src);
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_float(float2* dst, float2 src)
{
    *dst = src;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_float(float4* dst, float4 src)
{
    *dst = src;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_float(Float8_* dst, Float8_ src)
{
    *dst = src;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename A>
inline __device__ typename packed_type<float, num_elems<A>::value>::type convert_to_float(A u)
{
    return {};
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ float4 convert_to_float(float4 u)
{
    return u;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ float2 convert_to_float(float2 u)
{
    return u;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ float convert_to_float(float u)
{
    return u;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ Float8_ convert_to_float(uint4 u)
{
    Float8_ f8;
    f8.x = half2_to_float2(u.x);
    f8.y = half2_to_float2(u.y);
    f8.z = half2_to_float2(u.z);
    f8.w = half2_to_float2(u.w);
    return f8;
}

template<>
inline __device__ float2 convert_to_float(__nv_bfloat162 u)
{
    float2 ret = bf1622float2(u);
    return ret;
}

template<>
inline __device__ float4 convert_to_float(bf16_4_t u)
{
    float4 ret;
    float2 f2x = bf1622float2(u.x);
    float2 f2y = bf1622float2(u.y);
    ret.x      = f2x.x;
    ret.y      = f2x.y;
    ret.z      = f2y.x;
    ret.w      = f2y.y;
    return ret;
}

template<>
inline __device__ Float8_ convert_to_float(bf16_8_t u)
{
    Float8_ f8;
    f8.x = bf1622float2(u.x);
    f8.y = bf1622float2(u.y);
    f8.z = bf1622float2(u.z);
    f8.w = bf1622float2(u.w);
    return f8;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ float4 convert_to_float(uint2 u)
{
    float4 ret;
    float2 f2x = half2_to_float2(u.x);
    float2 f2y = half2_to_float2(u.y);
    ret.x      = f2x.x;
    ret.y      = f2x.y;
    ret.z      = f2y.x;
    ret.w      = f2y.y;
    return ret;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ float2 convert_to_float(uint32_t u)
{
    return half2_to_float2(u);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ int8_t cast_to_int8(float val)
{
    // union {
    //     int8_t  int8[2];
    //     int16_t int16;
    // };

    // asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=h"(int16) : "f"(val));
    // return int8[0];
    int ret;

    asm volatile("v_cvt_i32_f32 %0, %1 \n" : "=v"(ret) : "v"(val));

    // return __float2int_rn(val);
    return static_cast<int8_t>(ret);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ int32_t cast_to_int8(float2 val)
{
    union {
        int8_t  int8[2];
        int32_t int32;
    };

    int8[0] = cast_to_int8(val.x);
    int8[1] = cast_to_int8(val.y);
    return int32;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ int32_t cast_to_int8(float4 val)
{
    union {
        int8_t  int8[4];
        int32_t int32;
    };

    int8[0] = cast_to_int8(val.x);
    int8[1] = cast_to_int8(val.y);
    int8[2] = cast_to_int8(val.z);
    int8[3] = cast_to_int8(val.w);
    return int32;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ int64_t cast_to_int8(Float8_ val)
{
    union {
        int8_t  int8[8];
        int64_t int64;
    };

    int8[0] = cast_to_int8(val.x.x);
    int8[1] = cast_to_int8(val.x.y);
    int8[2] = cast_to_int8(val.y.x);
    int8[3] = cast_to_int8(val.y.y);
    int8[4] = cast_to_int8(val.z.x);
    int8[5] = cast_to_int8(val.z.y);
    int8[6] = cast_to_int8(val.w.x);
    int8[7] = cast_to_int8(val.w.y);
    return int64;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef ENABLE_FP8
inline __device__ void convert_to_fp8(__nv_fp8_e4m3* v, const __nv_bfloat16 u)
{
    v[0] = __nv_fp8_e4m3(u);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_to_fp8(fp8_2_t* v, const __nv_bfloat162 u)
{
    v[0] = fp8_2_t(u);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_to_fp8(fp8_4_t* v, const bf16_4_t u)
{
    reinterpret_cast<fp8_2_t*>(v)[0] = fp8_2_t(u.x);
    reinterpret_cast<fp8_2_t*>(v)[1] = fp8_2_t(u.y);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_to_fp8(fp8_8_t* v, const bf16_8_t u)
{
    v[0].x = fp8_2_t(u.x);
    v[0].y = fp8_2_t(u.y);
    v[0].z = fp8_2_t(u.z);
    v[0].w = fp8_2_t(u.w);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_to_fp8(__nv_fp8_e4m3* v, const uint16_t u)
{
    v[0] = __nv_fp8_e4m3(reinterpret_cast<const half&>(u));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_to_fp8(fp8_2_t* v, const uint32_t u)
{
    v[0] = fp8_2_t(reinterpret_cast<const half2&>(u));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_to_fp8(fp8_4_t* v, const uint2 u)
{
    union {
        uint2 u2;
        half2 h2[2];
    };

    u2 = u;

    reinterpret_cast<fp8_2_t*>(v)[0] = fp8_2_t(h2[0]);
    reinterpret_cast<fp8_2_t*>(v)[1] = fp8_2_t(h2[1]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_to_fp8(fp8_8_t* v, const uint4 u)
{
    union {
        uint4 u4;
        half2 h2[4];
    };

    u4 = u;

    v[0].x = fp8_2_t(h2[0]);
    v[0].y = fp8_2_t(h2[1]);
    v[0].z = fp8_2_t(h2[2]);
    v[0].w = fp8_2_t(h2[3]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_to_fp8(__nv_fp8_e4m3* v, const float u)
{
    v[0] = __nv_fp8_e4m3(u);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_to_fp8(fp8_2_t* v, const float2 u)
{
    v[0] = fp8_2_t(u);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_to_fp8(fp8_4_t* v, const float4 u)
{
    v[0] = fp8_4_t(u);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_to_fp8(fp8_8_t* v, const Float8_ u)
{
    v[0].x = fp8_2_t(u.x);
    v[0].y = fp8_2_t(u.y);
    v[0].z = fp8_2_t(u.z);
    v[0].w = fp8_2_t(u.w);
}
#endif  // ENABLE_FP8

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float float_from_int8(int8_t u)
{
    // printf("input int8 value: %d \n", u);
    return u;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float2 float_from_int8(int16_t u)
{
    union {
        int16_t int16;
        int8_t  int8[2];
    };

    int16 = u;
    // printf("input int8 value: %d %d\n", int8[0], int8[1]);
    return make_float2(int8[0], int8[1]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float4 float_from_int8(int32_t u)
{
    union {
        int32_t int32;
        int8_t  int8[4];
    };

    int32 = u;
    // printf("input int8 value: %d %d %d %d\n", int8[0], int8[1], int8[2], int8[3]);
    return make_float4(int8[0], int8[1], int8[2], int8[3]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// clang-format off
inline __device__ Float8_ float_from_int8(int64_t u)
{
    union {
        int64_t int64;
        int16_t int16[4];
    };
    int64 = u;
    return Float8_ {float_from_int8(int16[0]),
                    float_from_int8(int16[1]),
                    float_from_int8(int16[2]),
                    float_from_int8(int16[3])};
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, typename T_cache>
struct kv_cache_scale_type_t
{
    using Type = float;
};

#ifdef ENABLE_FP8
template <>
struct kv_cache_scale_type_t<half, __nv_fp8_e4m3>
{
    using Type = uint16_t;
};

template <>
struct kv_cache_scale_type_t<uint16_t, __nv_fp8_e4m3>
{
    using Type = uint16_t;
};

template <>
struct kv_cache_scale_type_t<__nv_bfloat16, __nv_fp8_e4m3>
{
    using Type = __nv_bfloat16;
};
#endif // ENALBE_FP8

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Vec_k, typename T, typename T_scale>
inline __device__ void load_8bits_kv_cache_vec(Vec_k* vec, const T* pointer, int idx, T_scale scale)
{
    assert(false); // Not used.
}

template <typename Vec_k, typename T, typename T_scale>
inline __device__ void store_8bits_kv_cache_vec(T* pointer, const Vec_k& vec, int idx, T_scale scale)
{
    assert(false); // Not used.
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Vec_k>
inline __device__ void load_8bits_kv_cache_vec(Vec_k* vec, const int8_t* pointer, int idx, float scale)
{
    using Packed_8bits_t = typename packed_type<int8_t, num_elems<Vec_k>::value>::type;
    using Packed_Float_t = typename packed_type<float, num_elems<Vec_k>::value>::type;
    const auto quant = *reinterpret_cast<const Packed_8bits_t*>(&pointer[idx]);

    convert_from_float(vec, mul<Packed_Float_t>(scale, float_from_int8(quant)));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef ENABLE_FP8
template <typename Vec_k, typename T_scale>
inline __device__ void load_8bits_kv_cache_vec(Vec_k* vec, const __nv_fp8_e4m3* pointer, int idx, T_scale scale)
{
    using Packed_8bits_t = typename packed_type<__nv_fp8_e4m3, num_elems<Vec_k>::value>::type;
    const auto quant = *reinterpret_cast<const Packed_8bits_t*>(&pointer[idx]);
    convert_from_fp8(vec, quant);
    vec[0] = mul<Vec_k>(scale, vec[0]);
}
#endif // ENABLE_FP8

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Vec_k>
inline __device__ void store_8bits_kv_cache_vec(int8_t* pointer, const Vec_k& vec, int idx, float scale)
{
    using Packed_8bits_t = typename packed_type<int8_t, num_elems<Vec_k>::value>::type;    // int32_t  int64_t
    using Packed_Float_t = typename packed_type<float, num_elems<Vec_k>::value>::type;     // float4   Float8_
    Packed_8bits_t out_quant = cast_to_int8(mul<Packed_Float_t>(scale, convert_to_float(vec)));
    *reinterpret_cast<Packed_8bits_t*>(&pointer[idx]) = out_quant;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef ENABLE_FP8
template <typename Vec_k, typename T_scale>
inline __device__ void store_8bits_kv_cache_vec(__nv_fp8_e4m3* pointer, const Vec_k& vec, int idx, T_scale scale)
{
    using Packed_8bits_t = typename packed_type<__nv_fp8_e4m3, num_elems<Vec_k>::value>::type;
    Packed_8bits_t out_quant;
    convert_to_fp8(&out_quant, mul<Vec_k>(scale, vec));

    *reinterpret_cast<Packed_8bits_t*>(&pointer[idx]) = out_quant;
}
#endif // ENABLE_FP8

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename I_T>
inline __device__ float fabs_max(const I_T a)
{
    return float{};  // for compile
}

template<>
inline __device__ float fabs_max(const float a) // for float
{
    return a;
}

template<>
inline __device__ float fabs_max(const float2 a) // for float
{
    float max_value = max(fabsf(a.x), fabsf(a.y));

    return max_value;
}

template<>
inline __device__ float fabs_max(const float4 a) // for float
{
    float max_value = max(fabsf(a.w), max(fabsf(a.z), max(fabsf(a.x), fabsf(a.y))));

    return max_value;
}

template<>
inline __device__ float fabs_max(const __nv_bfloat162 a) // for bf16
{
    float2 ele_x =  bf1622float2(a);
    float max_value = max(fabsf(ele_x.x), fabsf(ele_x.y));
    return max_value;
}

template<>
inline __device__ float fabs_max(const bf16_4_t a) // for bf16
{
    float2 ele_x =  bf1622float2(a.x);
    float2 ele_y =  bf1622float2(a.y);
    float max_x = max(fabsf(ele_x.x), fabsf(ele_x.y));
    float max_y = max(fabsf(ele_y.x), fabsf(ele_y.y));

    float max_value = max(max_x, max_y);
    return max_value;
}

template<>
inline __device__ float fabs_max(const bf16_8_t a) // for bf16
{
    float2 ele_x =  bf1622float2(a.x);
    float2 ele_y =  bf1622float2(a.y);
    float2 ele_z =  bf1622float2(a.z);
    float2 ele_w =  bf1622float2(a.w);
    float max_x = max(fabsf(ele_x.x), fabsf(ele_x.y));
    float max_y = max(fabsf(ele_y.x), fabsf(ele_y.y));
    float max_z = max(fabsf(ele_z.x), fabsf(ele_z.y));
    float max_w = max(fabsf(ele_w.x), fabsf(ele_w.y));

    float max_value = max(max_w, max(max_z, max(max_x, max_y)));

    return max_value;
}

template<>
inline __device__ float fabs_max(const uint32_t a) //for float16
{
    float2 ele_x = half2_to_float2(a);
    float max_value = max(fabsf(ele_x.x), fabsf(ele_x.y));
    return max_value;
}

template<>
inline __device__ float fabs_max(const uint2 a) //for float16
{
    float2 ele_x = half2_to_float2(a.x);
    float2 ele_y = half2_to_float2(a.y);
    float max_x = max(fabsf(ele_x.x), fabsf(ele_x.y));
    float max_y = max(fabsf(ele_y.x), fabsf(ele_y.y));

    float max_value = max(max_x, max_y);
    return max_value;
}

template<>
inline __device__ float fabs_max(const uint4 a) //for float16
{
    float2 ele_x = half2_to_float2(a.x);
    float2 ele_y = half2_to_float2(a.y);
    float2 ele_z = half2_to_float2(a.z);
    float2 ele_w = half2_to_float2(a.w);
    float max_x = max(fabsf(ele_x.x), fabsf(ele_x.y));
    float max_y = max(fabsf(ele_y.x), fabsf(ele_y.y));
    float max_z = max(fabsf(ele_z.x), fabsf(ele_z.y));
    float max_w = max(fabsf(ele_w.x), fabsf(ele_w.y));

    float max_value = max(max_w, max(max_z, max(max_x, max_y)));

    return max_value;
}

template <typename T>
inline __device__ __host__ T divUp(T m, T n)
{
    return (m + n - 1) / n;
}

template <typename T, int Dh, bool DO_CROSS_ATTENTION, bool SPLIT_KV_CACHE = false>
inline size_t multi_block_grid_setup(const Paged_multihead_attention_params<T, DO_CROSS_ATTENTION, SPLIT_KV_CACHE>& params,
    int threads_per_value, int threads_per_block, int tlength, bool do_multi_block)
{
    if (!do_multi_block)
    {
        return 1;
    }

    // auto constexpr threads_per_value = mmha::threads_per_value<T>(mmha::dh_max(Dh));
    int balanced_seq_len_tile
        = divUp(params.multi_processor_count, params.batch_size * params.num_heads);

    const int seq_len_per_kv_loop = mmha::divUp(threads_per_block , threads_per_value) * 1;
    int max_seq_len_tile = params.max_seq_len_tile;
    max_seq_len_tile = std::min(divUp(tlength + 1, seq_len_per_kv_loop), max_seq_len_tile);

    // A single CTA can at most compute 8k tokens due to lds size limit
    int min_seq_len_tile = divUp(tlength + 1, params.max_timesteps_per_block);

    // Make sure: seq_len_tile * threads_per_value <= threads_per_block (for multi_block_mode)
    params.seq_len_tile = std::clamp(balanced_seq_len_tile, min_seq_len_tile, max_seq_len_tile);

    // assert(params.seq_len_tile <= params.max_seq_len_tile);

    params.timesteps_per_block = divUp(tlength, params.seq_len_tile);

    params.enable_multi_block = (params.seq_len_tile > 1);
    // Return the sequence length tile if using multi block modes.
    return params.seq_len_tile;
}

}

