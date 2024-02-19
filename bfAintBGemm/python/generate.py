import argparse
import gemm_kernel_traits
import gemm_kernel_rr16r
import datatype

def gen_kernel_list():
    tile_b256_32x128x64 = gemm_kernel_traits.GemmTileSize(256, 32, 128, 64, 16, 32, 128, 32, 32, 8, 1, 8, 16, 8, 1, 8, 8, 8, 8, 8)
    k_list = []
    k_list.append(gemm_kernel_rr16r.GemmKernelRR16R(datatype.BF16, datatype.I8, datatype.BF16, datatype.F32, 1, tile_b256_32x128x64, "v1"))

    return k_list

def gen_list_blobs(kernel_list, list_blobs_path):
    pass

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

    write_and_compile_kernels(kernel_list, args.output_dir)

