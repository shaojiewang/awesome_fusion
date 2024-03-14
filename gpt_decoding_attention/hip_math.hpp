#include <hip/hip_runtime.h>

namespace math{

__device__ inline uint32_t next_power_of_two(uint32_t x)
{
    return 1 << (32 - __builtin_clz (x - 1));
}

}
