#pragma once

#include "hip_utils.h"
#include "hip_type_utils.cuh"

template <class TDst, class TSrc, class TSacle>
__global__ void matrix_elementwise_scale(TDst* dst, TSrc* src, TScale* scale, const int size)
{
}

template <class TDst, class TSrc, class TSacle>
void invokeMatrixElementwiseScale()
{
}

template void invokeMatrixElementwiseScale();
