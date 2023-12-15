#pragma once

#include <hip/hip_runtime.h>

struct SimpleMem
{
    SimpleMem() = default;
    SimpleMem(std::size_t mem_size) : p_mem_{} {};
    virtual void* GetBuffer() = 0;
    virtual ~SimpleMem() {}
    void* p_mem_;
};

struct SimpleDeviceMem : public SimpleMem
{
    SimpleDeviceMem() = delete;

    SimpleDeviceMem(std::size_t mem_size) : SimpleMem(mem_size)
    {
        (void)hipMalloc(static_cast<void**>(&p_mem_), mem_size);
    }

    void* GetBuffer() { return p_mem_; }

    ~SimpleDeviceMem() { (void)hipFree(p_mem_); }
};

struct SimpleHostMem : public SimpleMem
{
    SimpleHostMem() = delete;

    SimpleHostMem(std::size_t mem_size) : SimpleMem(mem_size)
    {
        p_mem_ = (void*)malloc(mem_size);
    }

    void* GetBuffer() { return p_mem_; }

    ~SimpleHostMem() { (void)free(p_mem_); }
};

