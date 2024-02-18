import argparse
import gemm_kernel_traits
import gemm_kernel_rr16r
import datatype


def write_kernels(output_dir):
    tile = gemm_kernel_traits.GemmTileSize(cta_size=256,
                                           cta_m=32,
                                           cta_n=128,
                                           cta_k=64,
                                           global_bk1=16,
                                           warp_m=32,
                                           warp_n=128,
                                           inst_m=32,
                                           inst_n=32,
                                           inst_k=8,
                                           inst_blocks=1,
                                           gmem_vec_a=8,
                                           gmem_vec_b=16,
                                           gmem_vec_c=8,
                                           gmem_vec_scale=1,
                                           smem_vec_a=8,
                                           smem_vec_b=8,
                                           smem_vec_c=8,
                                           smem_a_k1=8,
                                           smem_b_k1=8)
    k = gemm_kernel_rr16r.GemmKernelRR16R(datatype.BF16,
                                          datatype.I8,
                                          datatype.BF16,
                                          datatype.F32,
                                          1,
                                          tile,
                                          "v1")
    #print(k.a_layout)
    k.write_kernel(output_dir)

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

    write_kernels(args.output_dir)
