#pragma once

#include <assert.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <iostream>

#define CUSTOM_AR_SIZE_THRESHOLD 393216 // 50331648
#define MAX_ALL_REDUCE_BLOCKS 64
#define FLAG(a) ((uint32_t)((a)%0x146))
#define RANKS_PER_HIVE 4
#define HIVES_PER_NODE 2
#define RANKS_PER_NODE 8
#define WARP_SIZE 64
#define DEFAULT_BLOCK_SIZE 512
#define DEFALUT_ALGO_AR_SIZE_THRESHOLD_INTRA_HIVE (384 * 1024)
#define DEFALUT_ALGO_AR_SIZE_THRESHOLD_INTER_HIVE (32 * 1024)

namespace fastertransformer {

#ifdef ENABLE_BF16
typedef struct bf168 {
    __nv_bfloat162 x;
    __nv_bfloat162 y;
    __nv_bfloat162 z;
    __nv_bfloat162 w;
} bf168;
#endif

template<typename T>
struct AllReduceParams {
    size_t    elts_total;
    size_t    elts_per_rank;
    size_t    elts_per_block;
    size_t    rank_offset;
    size_t    rank, local_rank, node_id;
    uint32_t  barrier_flag;
    uint32_t* peer_barrier_ptrs[RANKS_PER_NODE];
    T*        peer_comm_buffer_ptrs[RANKS_PER_NODE];
    T*        local_output_buffer_ptr;
};

template<typename T>
void invokeOneOrTwoShotAllReduceKernel(AllReduceParams<T>& param, hipStream_t stream);

void kernelLaunchConfig(int& blocks_per_grid, int& threads_per_block, size_t elts, int kernel_algo, int tp_rank);

}  // namespace fastertransformer
