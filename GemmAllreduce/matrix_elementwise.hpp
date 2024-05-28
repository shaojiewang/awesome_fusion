#pragma once

#include "hip_utils.h"
#include "hip_type_utils.cuh"

namespace awesome_fusion {

template <class TDst, class TSrc, class TScale>
__global__ void matrix_elementwise_scale(TDst* dst, TSrc* src, TScale* scale, const int m, const int n)
{
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = tidx; i < m * n; i += blockDim.x * gridDim.x)
    {
        int i_scale = i / n;
        TScale res = src[i] * scale[i_scale];
        dst[i] = type_convert<TDst, TScale>(res);
    }
}

template <class TDst, class TSrc, class TScale>
void invokeMatrixElementwiseScale(TDst* dst, TSrc* src, TScale* scale, const int m, const int n)
{
    dim3 grids = {208};
    dim3 blocks = {512};
    matrix_elementwise_scale<<<grids, blocks>>>(dst, src, scale, m, n);
}

template void invokeMatrixElementwiseScale(hip_bfloat16* dst, int8_t* src, float* scale, const int m, const int n);

}
