#pragma once

#include <hip/hip_runtime.h>

#define GPU_CHECK_ERROR(call) do { \
    hipError_t err = call; \
    if(err != hipSuccess) { \
        printf("[hip error](%d), failed to call %s \n", (int)err, #call); \
        exit(0); \
    } \
} while(0) \

