#pragma once

#include "datatype.hpp"
#include "simple_device_mem.hpp"

template <typename Y, 
          typename X,
          typename YLocation,
          typename XLocation>
void mem_transfer(YLocation& dst, XLocation& src, std::size_t col, std::size_t row, std::size_t packed);

template<>
void mem_transfer<bfloat16, float, SimpleHostMem, SimpleHostMem>(
    SimpleHostMem& dst, 
    SimpleHostMem& src, 
    std::size_t col, 
    std::size_t row, 
    std::size_t packed)
{
    bfloat16* p_dst = (bfloat16*)(dst.GetBuffer());
    float* p_src = (float*)(src.GetBuffer());
    for(std::size_t i = 0; i < col; i++)
    {
        for(std::size_t j = 0; j < row; j++)
        {
            p_dst[i * row + j] = type_convert<bfloat16, float>(p_src[i * row + j]);
        }
    }
}

template<>
void mem_transfer<float, bfloat16, SimpleHostMem, SimpleHostMem>(
    SimpleHostMem& dst, 
    SimpleHostMem& src, 
    std::size_t col, 
    std::size_t row, 
    std::size_t packed)
{
    float* p_dst = (float*)(dst.GetBuffer());
    bfloat16* p_src = (bfloat16*)(src.GetBuffer());
    for(std::size_t i = 0; i < col; i++)
    {
        for(std::size_t j = 0; j < row; j++)
        {
            p_dst[i * row + j] = type_convert<float, bfloat16>(p_src[i * row + j]);
        }
    }
}

template<>
void mem_transfer<int8_t, float, SimpleHostMem, SimpleHostMem>(
    SimpleHostMem& dst, 
    SimpleHostMem& src, 
    std::size_t col, 
    std::size_t row, 
    std::size_t packed)
{
    int8_t* p_dst = (int8_t*)(dst.GetBuffer());
    float* p_src = (float*)(src.GetBuffer());
    for(std::size_t i = 0; i < col / packed; i++)
    {
        for(std::size_t j = 0; j < row; j++)
        {
            for(std::size_t k = 0; k < packed; k++)
            {
                p_dst[i * row * packed + j * packed + k] = type_convert<int8_t, float>(p_src[(i * packed + k) * row + j]) + 128;
            }
        }
    }
}

