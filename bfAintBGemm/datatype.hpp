#pragma once
#include <hip/hip_fp16.h>

using float16 = __half;
using bfloat16 = unsigned short;

template <typename Y,
          typename X,
          std::enable_if_t<!(std::is_const_v<Y> || std::is_const_v<X>), bool> = false>
__host__ __device__ constexpr Y type_convert(X x) {
     static_assert(!std::is_reference_v<Y> && !std::is_reference_v<X>);

    return static_cast<Y>(x);
}



