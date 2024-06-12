#!/bin/bash
mpirun -n 4 --allow-run-as-root build/test_gemm_ar 1 2048 256 4 0

