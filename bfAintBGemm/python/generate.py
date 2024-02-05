import argparse
import gemm_kerel_rr16r


def write_kernels():
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='kernel_gen',
        description='generate asm kernel for high perf operators on rocm',
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        required=False,
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

