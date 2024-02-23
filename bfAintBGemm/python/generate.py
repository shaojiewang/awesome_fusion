import os
import argparse

import gemm_kernel_traits
import gemm_kernel_rr16r
import datatype
import kernel_list_header
import host_side_compile

def gen_kernel_list():
    # tile_b256_32x128x64 = gemm_kernel_traits.GemmTileSize(256, 32, 128, 64, 16, 32, 128, 32, 32, 8, 1, 8, 16, 8, 1, 8, 8, 8, 8, 8)
    tile_b256_128x128x32 = gemm_kernel_traits.GemmTileSize(256, 128, 128, 32, 16, 64, 64, 32, 32, 8, 1, 8, 16, 8, 1, 8, 8, 8, 8, 8)
    k_list = []
    k_list.append(gemm_kernel_rr16r.GemmKernelRR16R(datatype.BF16, datatype.I8, datatype.BF16, datatype.F32, datatype.F32, datatype.BF16, 1, tile_b256_32x128x64, "v1"))

    return k_list

def gen_list_blobs(kernel_list, list_blobs_path):
    k_list_header = kernel_list_header.KernelListHeader(kernel_list)
    k_list_header.write_header(list_blobs_path)
    
def host_compile(host_code_path, exe_path, tmp_path):
    host_side_obj = host_side_compile.HostSideCompile(host_code_path, exe_path, tmp_path)
    host_side_obj.compile_host()


def write_and_compile_kernels(k_list, output_dir):
    #print(k.a_layout)
    for k in k_list:
        k.write_kernel(output_dir)
        k.compile_kernel(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='kernel_gen',
        description='generate asm kernel for high perf operators on rocm',
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        required=True,
        help="write kernels into a directory",
    )

    parser.add_argument(
        "-l",
        "--list_blobs",
        required=False,
        help="list all kernel to a file"
    )

    args = parser.parse_args()

    print(f"ouput dir={args.output_dir}, list={args.list_blobs}")

    # blob list
    kernel_list = gen_kernel_list()
    gen_list_blobs(kernel_list, args.list_blobs)

    # kernel code writer and compile
    write_and_compile_kernels(kernel_list, args.output_dir)

    # host code compile
    host_code = 'bfAintBGemm.cc'
    exe_name = 'bfAintBGemm.exe'
    exe_path = os.path.join(args.output_dir, exe_name)
    host_compile(host_code, exe_path, args.output_dir)

