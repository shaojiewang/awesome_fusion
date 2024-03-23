
#pragma once

#define B_PACKED_K 16

struct kernel_tunable {
    int wg_size;
    int wg_tile_m;
    int wg_tile_n;
    int wg_tile_k;
    int b_packed_k;
    std::string kernel_name;
};

static inline std::vector<kernel_tunable> 
get_kernel_list() {
    std::vector<kernel_tunable> k_list = {
        kernel_tunable{256, 128, 128, 32, B_PACKED_K, "bf16gemm_rr16r_b256_128x128x32_wg2x2_w2x2_32x32x8bf16_1k_pregld1_pipeline_interleaved_splitk"}, 

    };
    return k_list;
}

