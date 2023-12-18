#include <stdio.h>
#include <hip/hip_runtime.h>
#include <random>
#include <iostream>
#include <stdlib.h>
#include <math.h>

#include "simple_device_mem.hpp"
#include "gemm_tensor_layout.hpp"
#include "datatype.hpp"
#include "host_ref.hpp"
#include "validation.hpp"
#include "random_gen.hpp"
#include "mem_transfer.hpp"
#include "gpu_utils.hpp"

using Row = gemm_layout::gemm::RowMajor;
using Col = gemm_layout::gemm::ColumnMajor;

using ALayout = Row;
using BLayout = Row;
using ScaleLayout = Row;
using CLayout = Row;

using ADataType = bfloat16;
using BDataType = int8_t;
using ScaleDataType = float;
using CDataType = bfloat16;

#define HSACO "bf16gemm_kernel_gfx90a.hsaco"
#define KER_NAME "bf16gemm_rrr"

int main(int argc, char ** argv)
{
    int validation = 0;
    int m = 32;
    int n = 64;
    int k = 128;
    if(argc >= 2) {
        validation = atoi(argv[1]);
    }
    if(argc >= 5) {
        m = atoi(argv[2]);
        n = atoi(argv[3]);
        k = atoi(argv[4]);
    }
    int lda = k;
    int ldb = n;
    int ldc = n;

    if(argc >= 8) {
        lda = atoi(argv[5]);
        ldb = atoi(argv[6]);
        ldc = atoi(argv[7]);
    }

    auto f_matrix_space_size = 
        [](std::size_t nRow, std::size_t nCol, std::size_t stride, auto layout){
            using Layout = decltype(layout);
            if constexpr(std::is_same<Layout, Row>::value) {
                return (nRow - 1) * stride + nCol;
            } else {
                return (nCol - 1) * stride + nRow;
            }
        };

    SimpleDeviceMem a_device_buf(sizeof(ADataType) * f_matrix_space_size(m, k, lda, ALayout{}));
    SimpleDeviceMem b_device_buf(sizeof(BDataType) * f_matrix_space_size(k, n, ldb, BLayout{}));
    SimpleDeviceMem c_device_buf(sizeof(CDataType) * f_matrix_space_size(m, n, ldc, CLayout{}));
    SimpleDeviceMem scale_device_buf(sizeof(ScaleDataType) * f_matrix_space_size(n, 1, 1, ScaleLayout{}));
    
    SimpleHostMem a_host_buf(sizeof(float) * f_matrix_space_size(m, k, lda, ALayout{}));
    SimpleHostMem b_host_buf(sizeof(float) * f_matrix_space_size(k, n, ldb, BLayout{}));
    SimpleHostMem c_host_buf(sizeof(float) * f_matrix_space_size(m, n, ldc, CLayout{}));
    SimpleHostMem scale_host_buf(sizeof(float) * f_matrix_space_size(n, 1, 1, ScaleLayout{}));

    rand_vector_2d_int(reinterpret_cast<float*>(a_host_buf.GetBuffer()), m, k, lda);
    rand_vector_2d_int(reinterpret_cast<float*>(b_host_buf.GetBuffer()), n, k, ldb);
    rand_vector_2d_int(reinterpret_cast<float*>(scale_host_buf.GetBuffer()), n, 1, 1);

    SimpleHostMem a_host_buf_to_device(sizeof(ADataType) * f_matrix_space_size(m, k, lda, ALayout{}));
    SimpleHostMem b_host_buf_to_device(sizeof(BDataType) * f_matrix_space_size(k, n, ldb, BLayout{}));
    SimpleHostMem c_host_buf_from_device(sizeof(CDataType) * f_matrix_space_size(m, n, ldc, CLayout{}));

    mem_transfer<ADataType, float, SimpleHostMem, SimpleHostMem>(a_host_buf_to_device, a_host_buf, m * k);
    mem_transfer<BDataType, float, SimpleHostMem, SimpleHostMem>(b_host_buf_to_device, b_host_buf, n * k);

    hipMemcpy(a_device_buf.GetBuffer(), a_host_buf_to_device.GetBuffer(), m * k * sizeof(ADataType), hipMemcpyHostToDevice);
    hipMemcpy(b_device_buf.GetBuffer(), b_host_buf_to_device.GetBuffer(), n * k * sizeof(BDataType), hipMemcpyHostToDevice);
    hipMemcpy(scale_device_buf.GetBuffer(), scale_host_buf.GetBuffer(), n * 1 * sizeof(ScaleDataType), hipMemcpyHostToDevice);

    int total_loop=10;
    int warm_ups = 10;
    int i;

    int bdx = 256;
    int gdx = ((m + 31) >> 5 ) * ((n + 63) >> 6);

// TODO: move this section to a header file

#ifdef ASM_PRINT
    //debug pointer
    float *host_print, *print;
    host_print = (float*)malloc(bdx*8);
    GPU_CHECK_ERROR(hipMalloc(&print, bdx*8));
#endif
    struct __attribute__((packed)) {
        void*  ptr_c;
        void*  ptr_a;
        void*  ptr_b;
        void*  ptr_scale
        unsigned int m;
        unsigned int n;
        unsigned int k;
        unsigned int lda;
        unsigned int ldb;
        unsigned int ldc;
        #ifdef ASM_PRINT
        void*  print;
        #endif
    } args;
    size_t arg_size = sizeof(args);
    args.ptr_c  = c_device_buf.GetBuffer();
    args.ptr_a  = a_device_buf.GetBuffer();
    args.ptr_b  = b_device_buf.GetBuffer();
    args.ptr_scale  = scale_device_buf.GetBuffer();
    args.m      = m;
    args.n      = n;
    args.k      = k;
    args.lda    = lda;
    args.ldb    = ldb;
    args.ldc    = ldc;
    #ifdef ASM_PRINT
    args.print  = (void*)print;
    #endif
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE,
                    &arg_size, HIP_LAUNCH_PARAM_END};
    
    for(i=0;i<warm_ups;i++){
        GPU_CHECK_ERROR(hipModuleLaunchKernel(kernel_func, gdx,1,1, bdx,1,1,  0, 0, NULL, (void**)&config ));
        //std::cout<<"safe here"<<std::endl;
    }

#ifdef ASM_PRINT
    int max_i=256;
    GPU_CHECK_ERROR(hipMemcpy(host_print, print, 8*max_i, hipMemcpyDeviceToHost));
    for(int i=0; i<max_i; i++){
        if(((uint32_t*)host_print)[2*i+1]!=0x5c005c00)
        printf("Thread%d, PrintVal:0x%x\n",((int*) host_print)[2*i], ((uint32_t*)host_print)[2*i+1]);
        //std::cout<<"Thread"<<((int*) host_print)[2*i]<<", PrintVal1:"<<(((float16*)host_print)[4*i+2])<<
        //", PrintVal2:"<<( ( (float16*)host_print )[4*i+3] )<<std::endl;
    }    
#endif

    hipEventCreate(&evt_00);
    hipEventCreate(&evt_11);
    hipDeviceSynchronize();
    hipEventRecord(evt_00, NULL);
    for(i=0;i<total_loop;i++)
        GPU_CHECK_ERROR(hipModuleLaunchKernel(kernel_func, gdx,1,1, bdx,1,1,  0, 0, NULL, (void**)&config ));

    float elapsed_ms;
    hipEventRecord(evt_11, NULL);
    hipEventSynchronize(evt_11);
    hipDeviceSynchronize();
    hipEventElapsedTime(&elapsed_ms, evt_00, evt_11);
    hipEventDestroy(evt_00);
    hipEventDestroy(evt_11);

    float time_per_loop = elapsed_ms/total_loop;
    float gflops = (float)2*m*n*k/time_per_loop/(1e6);
    printf("m:%d,n:%d,k:%d,gflops:%.3f\n",m,n,k,gflops);
    printf("\n");

    if(validation)
    {
        gemm_rrr(reinterpret_cast<float*>(c_host_buf.GetBuffer()),
                 reinterpret_cast<float*>(a_host_buf.GetBuffer()),
                 reinterpret_cast<float*>(b_host_buf.GetBuffer()),
                 reinterpret_cast<float*>(scale_host_buf.GetBuffer()),
                 m, 
                 n,
                 k,
                 k,
                 n, 
                 n);
        
        GPU_CHECK_ERROR(hipMemcpy(c_host_buf_from_device.GetBuffer(), c_device_buf.GetBuffer(), ldc * n * sizeof(CDataType), hipMemcpyDeviceToHost));
        bool res = valid_vector(c_host_buf.GetBuffer(), c_host_buf_from_device.GetBuffer(),  m * n);
        printf(",%s", res ? "valid" : "fail");
    }
    
    return 0;
}
