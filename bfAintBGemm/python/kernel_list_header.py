import os


class KernelListHeader(object):
    def __init__(self, kernel_list):
        self.kernel_list = kernel_list
        self.header_name = "kernel_list.hpp"
        self.header_str = self.gen_header_code()

    def gen_header_code(self):
        HEADER = """
#pragma once

#define B_PACKED_K {F_b_packed_k}

struct kernel_tunable {{
    int wg_size;
    int wg_tile_m;
    int wg_tile_n;
    int wg_tile_k;
    int b_packed_k;
    std::string kernel_name;
}};

static inline std::vector<kernel_tunable> 
get_kernel_list() {{
    std::vector<kernel_tunable> k_list = {{
        {F_kerel_list}
    }};
    return k_list;
}}

"""
        TUNABLE = "kernel_tunable{{{F_wg_size}, {F_wg_tile_m}, {F_wg_tile_n}, {F_wg_tile_k}, B_PACKED_K, \"{F_name}\"}}, \n"
        str_tunable = ""
        for k in self.kernel_list:
            str_tunable += TUNABLE.format(
                F_wg_size=k.tile.cta_size,
                F_wg_tile_m=k.tile.cta_m,
                F_wg_tile_n=k.tile.cta_n,
                F_wg_tile_k=k.tile.cta_k,
                F_name=k.get_kernel_name()
            )

        str_header = HEADER.format(F_b_packed_k=k.tile.global_bk1, F_kerel_list=str_tunable)

        return str_header

    def write_header(self, list_blobs_path):
        if os.path.exists(list_blobs_path):
            h_path = os.path.join(list_blobs_path, self.header_name)
            with open(h_path, "w") as f_header:
                f_header.write(self.header_str)
        else:
            assert false, "{}, path not exist".format(list_blobs_path)

