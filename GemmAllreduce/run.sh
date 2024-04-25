#!/bin/bash
hipcc custom_ar_kernels.cu custom_ar_comm.cc nccl.cc test.cc -I/usr/local/include -pthread -L/usr/local/lib -L /opt/rocm/rccl/lib/ -Wl,-rpath -Wl,/usr/local/lib -Wl,--enable-new-dtags -lmpi -lrccl -std=c++14 -DBUILD_MULTI_GPU=ON --offload-arch=gfx90a
mpirun -np 4 --allow-run-as-root ./a.out 

