#!/bin/bash
mpirun -n 4 --allow-run-as-root build/test_gemm_ar 1 8192 8192 4 0
mpirun -n 4 --allow-run-as-root build/test_gemm_ar 16 8192 8192 4 0
# mpirun -n 4 --allow-run-as-root build/test_gemm_ar 32 8192 8192 4 0

