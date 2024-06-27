#include <hip/hip_runtime.h>

__global__ void test_syncthreads(int* gpu_mem)
{
    int tidx = threadIdx.x;
    int last_block = 0;
    if (tidx == 0)
    {
        last_block = 1;
    }

    if (__syncthreads_or(last_block))
    {
        gpu_mem[tidx] = last_block;
    }

}

void invoke_test_kernel(int* gpu_mem)
{
    test_syncthreads<<<1, 256, 0, 0>>>(gpu_mem);
}

