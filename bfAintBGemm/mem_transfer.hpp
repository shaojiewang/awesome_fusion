#pragma once

#include "datatype.hpp"
#include "simple_device_mem.hpp"

template <typename Y, 
          typename X,
          typename YLocation,
          typename XLocation>
void mem_transfer(YLocation& dst, XLocation& src, std::size_t size);

template<>
void mem_transfer<bfloat16, float, SimpleHostMem, SimpleHostMem>(SimpleHostMem& dst, SimpleHostMem& src, std::size_t size)
{
    bfloat16* p_dst = (bfloat16*)(dst.GetBuffer());
    float* p_src = (float*)(src.GetBuffer());
    for(std::size_t i = 0; i < size; i++)
    {
        p_dst[i] = type_convert<bfloat16, float>(p_src[i]);
    }
}

template<>
void mem_transfer<int8_t, float, SimpleHostMem, SimpleHostMem>(SimpleHostMem& dst, SimpleHostMem& src, std::size_t size)
{
    int8_t* p_dst = (int8_t*)(dst.GetBuffer());
    float* p_src = (float*)(src.GetBuffer());
    for(std::size_t i = 0; i < size; i++)
    {
        p_dst[i] = type_convert<int8_t, float>(p_src[i]) + 128;
    }
}

