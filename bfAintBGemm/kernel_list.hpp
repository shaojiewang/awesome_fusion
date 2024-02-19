#pragma once

struct kernel_tunable {
    int wg_size;
    int wg_tile_m;
    int wg_tile_n;
    int wg_tile_k;
    int b_packed_k;
    std::string kernel_name;
}

static inline std::vector<kernel_tunable> 
get_kernel_list() {
    std::vector<kernel_tunable> k_list = {
        kernel_tunable{256, 32, 128, 64, 16, "bf16gemm_rr16r_b256_32x128x64_wg1x1_w1x4_32x32x8bf16_1k_pregld1_pipelined_splitk"}
    };
}

