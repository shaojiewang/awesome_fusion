
#pragma once

#ifdef ENABLE_BF16
#if defined(__CUDACC__)
#include <cuda_bf16.h>
#else

struct bfloat16_t
{
    short x;
#if 0
    __host__ __device__ bfloat16_t() : x(0) {} 
    __host__ __device__ bfloat16_t(float val) 
    {
        int val_i = __builtin_bit_cast(int, val);
        x = (short)(val_i >> 16);
    }
#endif 
};

using __nv_bfloat16 = bfloat16_t;
using bhalf_t = bfloat16_t;

// vector_type
template <typename T, int N>
struct vector_type;

template <typename T>
struct vector_type<T, 2>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));

    using type = d2_t;

    union {
        d2_t d2_;
    } data_;

    __host__ __device__ vector_type() : data_(type{0}) {}
    __host__ __device__ vector_type(type v) : data_(v) {}

};

template <typename T>
struct vector_type<T, 4>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    using type = d4_t;

    union {
        d4_t d4_;
    } data_;
    
    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}
};

template <typename T>
struct vector_type<T, 8>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    using type = d8_t;

    union {
        d8_t d8_;
    } data_;
    
    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}
};

// TODO: tmp solution of bf162 for ft code
#if defined (__HIPCC__)
struct __nv_bfloat162 
{
    __nv_bfloat16 x;
    __nv_bfloat16 y;
};

struct bf16_4_t {
    __nv_bfloat162 x;
    __nv_bfloat162 y;
};

struct bf16_8_t {
    __nv_bfloat162 x;
    __nv_bfloat162 y;
    __nv_bfloat162 z;
    __nv_bfloat162 w;
};

// Declare a template function for bf16 conversion using RTN
template <typename Y, typename X>
__host__ __device__ constexpr Y bf16_convert_rtn(X x);

// Convert fp32 to bf16 with RTN if higher precision is needed
template <>
inline __host__ __device__ constexpr bhalf_t bf16_convert_rtn<bhalf_t, float>(float x)
{
    union
    {
        float fp32;
        uint32_t int32;
    } u = {x};

    // When the exponent bits are not all 1s, then the value is zero, normal,
    // or subnormal. We round the bfloat16 mantissa up by adding 0x7FFF, plus
    // 1 if the least significant bit of the bfloat16 mantissa is 1 (odd).
    // This causes the bfloat16's mantissa to be incremented by 1 if the 16
    // least significant bits of the float mantissa are greater than 0x8000,
    // or if they are equal to 0x8000 and the least significant bit of the
    // bfloat16 mantissa is 1 (odd). This causes it to be rounded to even when
    // the lower 16 bits are exactly 0x8000. If the bfloat16 mantissa already
    // has the value 0x7f, then incrementing it causes it to become 0x00 and
    // the exponent is incremented by one, which is the next higher FP value
    // to the unrounded bfloat16 value. When the bfloat16 value is subnormal
    // with an exponent of 0x00 and a mantissa of 0x7f, it may be rounded up
    // to a normal value with an exponent of 0x01 and a mantissa of 0x00.
    // When the bfloat16 value has an exponent of 0xFE and a mantissa of 0x7F,
    // incrementing it causes it to become an exponent of 0xFF and a mantissa
    // of 0x00, which is Inf, the next higher value to the unrounded value.
    bool flag0 = ~u.int32 & 0x7f800000;

    // When all of the exponent bits are 1, the value is Inf or NaN.
    // Inf is indicated by a zero mantissa. NaN is indicated by any nonzero
    // mantissa bit. Quiet NaN is indicated by the most significant mantissa
    // bit being 1. Signaling NaN is indicated by the most significant
    // mantissa bit being 0 but some other bit(s) being 1. If any of the
    // lower 16 bits of the mantissa are 1, we set the least significant bit
    // of the bfloat16 mantissa, in order to preserve signaling NaN in case
    // the bfloat16's mantissa bits are all 0.
    bool flag1 = !flag0 && (u.int32 & 0xffff);

    u.int32 += flag0 ? 0x7fff + ((u.int32 >> 16) & 1) : 0; // Round to nearest, round to even
    u.int32 |= flag1 ? 0x10000 : 0x0;                      // Preserve signaling NaN

    return __builtin_bit_cast(bhalf_t, uint16_t(u.int32 >> 16));
}

// Convert X to Y
template <typename Y, typename X>
__host__ __device__ constexpr Y type_convert(X x)
{
    static_assert(!std::is_reference_v<Y> && !std::is_reference_v<X>);

    return static_cast<Y>(x);
}

// convert bfp16 to fp32
template <>
inline __host__ __device__ constexpr float type_convert<float, bhalf_t>(bhalf_t x)
{
    union
    {
        uint32_t int32;
        float fp32;
    } u = {uint32_t(x.x) << 16};

    return u.fp32;
}

// convert fp32 to bf16
template <>
inline __host__ __device__ constexpr bhalf_t type_convert<bhalf_t, float>(float x)
{
    int val_i = __builtin_bit_cast(int, x);
    return __builtin_bit_cast(bhalf_t, uint16_t(val_i >> 16));
}

#if defined(__HIPCC__)
inline __device__ float __bfloat162float(const __nv_bfloat16 x)
{
	return type_convert<float, bhalf_t>(x);
}

inline __device__ __nv_bfloat16 __float2bfloat16(float x)
{
	return bf16_convert_rtn<__nv_bfloat16, float>(x);
}

inline __device__ __nv_bfloat162 bf162bf162(const __nv_bfloat16 val) {
    __nv_bfloat162 val2;
    val2.x = val;
    val2.y = val;
    return val2;
}

inline __device__ __nv_bfloat162 __floats2bfloat162_rn(const float x, const float y)
{
	__nv_bfloat162 ret;
	ret.x = bf16_convert_rtn<__nv_bfloat16, float>(x);
	ret.y = bf16_convert_rtn<__nv_bfloat16, float>(y);
	return ret;
}

inline __device__ __nv_bfloat162 __floats2bfloat162_rn(const float2 x)
{
	__nv_bfloat162 ret;
	ret.x = bf16_convert_rtn<__nv_bfloat16, float>(x.x);
	ret.y = bf16_convert_rtn<__nv_bfloat16, float>(x.y);
	return ret;
}
#endif

inline __device__ float2 bf1622float2(const __nv_bfloat162 val) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800) || defined(__HIPCC__) 
    float2 f_val;
    f_val.x = type_convert<float, bhalf_t>(val.x); 
    f_val.y = type_convert<float, bhalf_t>(val.y);
    return f_val;
#else
    return __bfloat1622float2(val);
#endif
}

inline __device__ __nv_bfloat16 bf16hadd(const __nv_bfloat16 x, const __nv_bfloat16 y) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800) || defined(__HIPCC__)
    return __float2bfloat16( __bfloat162float(x) + __bfloat162float(y) );
#else
    return __hadd(x, y);
#endif
}

inline __device__ __nv_bfloat162 bf16hadd2(const __nv_bfloat162 x, const __nv_bfloat162 y) {
    float2 ret;
    ret.x = __bfloat162float(x.x) + __bfloat162float(y.x);
    ret.y = __bfloat162float(x.y) + __bfloat162float(y.y);
    return __floats2bfloat162_rn(ret);
}

inline __device__ __nv_bfloat162 make_bfloat162(const __nv_bfloat16 x, const __nv_bfloat16 y)
{
    __nv_bfloat162 t; t.x = x; t.y = y; return t;
}

inline __device__ __nv_bfloat162 bf16hsub2(const __nv_bfloat162 x, const __nv_bfloat162 y) {
    float2 ret;
    ret.x = __bfloat162float(x.x) - __bfloat162float(y.x);
    ret.y = __bfloat162float(x.y) - __bfloat162float(y.y);
    return __floats2bfloat162_rn(ret);
}

inline __device__ __nv_bfloat16 bf16hsub(const __nv_bfloat16 x, const __nv_bfloat16 y) {
    return __float2bfloat16( __bfloat162float(x) - __bfloat162float(y) );
}

inline __device__ __nv_bfloat162 bf16hmul2(const __nv_bfloat162 x, const __nv_bfloat162 y) {
    float2 ret;
    ret.x = __bfloat162float(x.x) * __bfloat162float(y.x);
    ret.y = __bfloat162float(x.y) * __bfloat162float(y.y);
    return __floats2bfloat162_rn(ret);
}

inline __device__ __nv_bfloat16 bf16hmul(const __nv_bfloat16 x, const __nv_bfloat16 y) {
    return __float2bfloat16( __bfloat162float(x) * __bfloat162float(y) );
}

inline __device__ __nv_bfloat16 bf16abs(const __nv_bfloat16 x)
{
	return __builtin_bit_cast(__nv_bfloat16, (short)(x.x | 0x8000));
}

inline __device__ __nv_bfloat162 bf16abs2(const __nv_bfloat162 x)
{
	return make_bfloat162(bf16abs(x.x), bf16abs(x.y));
}

inline __device__ __nv_bfloat162 operator*(const __nv_bfloat162 x, const __nv_bfloat162 y) { return bf16hmul2(x, y); };
inline __device__ __nv_bfloat162 operator+(const __nv_bfloat162 x, const __nv_bfloat162 y) { return bf16hadd2(x, y); };

inline __device__ __nv_bfloat162 bf16hfma2(const __nv_bfloat162 x, const __nv_bfloat162 y, const __nv_bfloat162 z) {
    float2 ret;
    ret.x = __bfloat162float(x.x) * __bfloat162float(y.x) + __bfloat162float(z.x);
    ret.y = __bfloat162float(x.y) * __bfloat162float(y.y) + __bfloat162float(z.y);
    return __floats2bfloat162_rn(ret);
}

inline __device__ __nv_bfloat16 bf16hfma(const __nv_bfloat16 x, const __nv_bfloat16 y, const __nv_bfloat16 z) {
    return __float2bfloat16( __bfloat162float(x) * __bfloat162float(y) + __bfloat162float(z));
}

inline __device__ __nv_bfloat162 bf16exp2(const __nv_bfloat162 x) {
    float2 ret;
    ret.x = expf(__bfloat162float(x.x));
    ret.y = expf(__bfloat162float(x.y));;
    return __floats2bfloat162_rn(ret);
}

inline __device__ __nv_bfloat16 bf16hadd(__nv_bfloat16 a, __nv_bfloat16 b, __nv_bfloat16 c) {
    return __float2bfloat16(__bfloat162float(a) + __bfloat162float(b) + __bfloat162float(c));
}

inline __device__ __nv_bfloat16 bf16hadd(__nv_bfloat16 a, __nv_bfloat16 b, __nv_bfloat16 c, __nv_bfloat16 d) {
    return __float2bfloat16(__bfloat162float(a) + __bfloat162float(b) + __bfloat162float(c) + __bfloat162float(d));
}

inline __device__ __nv_bfloat162 bf16hadd2(__nv_bfloat162 a, __nv_bfloat162 b, __nv_bfloat162 c) {
    float2 ret;
    ret.x = __bfloat162float(a.x) + __bfloat162float(b.x) + __bfloat162float(c.x);
    ret.y = __bfloat162float(a.y) + __bfloat162float(b.y) + __bfloat162float(c.y);
    return __floats2bfloat162_rn(ret);
}

inline __device__ __nv_bfloat16 bf16hmul(__nv_bfloat16 a, __nv_bfloat16 b, __nv_bfloat16 c) {
    return __float2bfloat16(__bfloat162float(a) * __bfloat162float(b) * __bfloat162float(c));
}

inline __device__ __nv_bfloat162 bf16hmul2(__nv_bfloat162 a, __nv_bfloat162 b, __nv_bfloat162 c) {
    float2 ret;
    ret.x = __bfloat162float(a.x) * __bfloat162float(b.x) * __bfloat162float(c.x);
    ret.y = __bfloat162float(a.y) * __bfloat162float(b.y) * __bfloat162float(c.y);
    return __floats2bfloat162_rn(ret);
}

inline __device__ __nv_bfloat162 bf16hfma2(__nv_bfloat162 a, __nv_bfloat162 b, __nv_bfloat162 c, __nv_bfloat162 d) {
    float2 ret;
    ret.x = __bfloat162float(a.x) * __bfloat162float(b.x) * __bfloat162float(c.x) + __bfloat162float(d.x);
    ret.y = __bfloat162float(a.y) * __bfloat162float(b.y) * __bfloat162float(c.y) + __bfloat162float(d.y);
    return __floats2bfloat162_rn(ret);
}

#endif
#endif
#endif

