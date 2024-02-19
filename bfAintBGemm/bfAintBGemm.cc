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
#include "tensor_reduction.hpp"

#include "build/kernel_list.hpp"

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
// #define KER_NAME "bf16gemm_rr8r_wg512_32x64x64_wg1x1_w2x4_16x16x16bf16_1k_pregld2"
// #define KER_NAME "bf16gemm_rr16r_b256_32x128x64_wg1x1_w1x4_32x32x8bf16_1k_pregld1_pipelined_splitk"
// #define KER_NAME "bf16gemm_rr16r_b256_32x128x64_wg1x1_w1x4_32x32x8bf16_1k_pregld1"
// #define KER_NAME "bf16gemm_rr8r_b256_32x128x64_wg1x1_w1x4_32x32x8bf16_1k_pregld1"
// #define KER_NAME "bf16gemm_rr8r_wg512_32x64x64_wg1x1_w2x4_16x16x16bf16_1k_pregld1"
// #define KER_NAME "bf16gemm_rr8r_wg128_32x64x64_wg1x1_w1x2_32x32x8bf16_1k_pregld1"
// #define KER_NAME "bf16gemm_rrr_wg256_32x256x64_wg1x2_w1x4_32x32x8bf16_1k_pregld1"
// #define KER_NAME "bf16gemm_rrr_wg256_32x128x64_wg1x1_w1x4_32x32x8bf16_1k_pregld1"
// #define KER_NAME "bf16gemm_rrr_wg1x1_w1x2_32x32x8bf16_1k_pregld1"
// #define KER_NAME "bf16gemm_rrr_wg1x1_w1x2_32x32x8bf16_1k_pregld2"


int main(int argc, char ** argv)
{
    int validation = 0;
    int m = 32;
    int n = 64 * 2;
    int k = 256 * 2;
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

    // get kernel list
    std::vector<kernel_tunable> k_list = get_kernel_list();

    hipModule_t module;
    hipFunction_t kernel_func;
    hipEvent_t evt_00, evt_11;
    GPU_CHECK_ERROR(hipSetDevice(0));

    auto f_matrix_space_size = 
        [](std::size_t nRow, std::size_t nCol, std::size_t stride, auto layout){
            using Layout = decltype(layout);
            if constexpr(std::is_same<Layout, Row>::value) {
                return (nRow - 1) * stride + nCol;
            } else {
                return (nCol - 1) * stride + nRow;
            }
        };

    int sk_blocks = 2;
    
    SimpleDeviceMem a_device_buf(sizeof(ADataType) * f_matrix_space_size(m, k, lda, ALayout{}));
    SimpleDeviceMem b_device_buf(sizeof(BDataType) * f_matrix_space_size(k, n, ldb, BLayout{}));
    SimpleDeviceMem c_device_buf(sizeof(CDataType) * f_matrix_space_size(m, n, ldc, CLayout{}));
    SimpleDeviceMem c_workspace_device_buf(sizeof(CDataType) * f_matrix_space_size(m * sk_blocks, n, ldc, CLayout{}));
    SimpleDeviceMem scale_device_buf(sizeof(ScaleDataType) * f_matrix_space_size(n, 1, 1, ScaleLayout{}));
    
    SimpleHostMem a_host_buf(sizeof(float) * f_matrix_space_size(m, k, lda, ALayout{}));
    SimpleHostMem b_host_buf(sizeof(float) * f_matrix_space_size(k, n, ldb, BLayout{}));
    SimpleHostMem c_host_buf(sizeof(float) * f_matrix_space_size(m, n, ldc, CLayout{}));
    SimpleHostMem scale_host_buf(sizeof(float) * f_matrix_space_size(n, 1, 1, ScaleLayout{}));

    rand_vector_2d_int_a(reinterpret_cast<float*>(a_host_buf.GetBuffer()), m, k, lda);
    rand_vector_2d_int_b(reinterpret_cast<float*>(b_host_buf.GetBuffer()), k, n, ldb);
    rand_vector_2d_int_scale(reinterpret_cast<float*>(scale_host_buf.GetBuffer()), n, 1, 1);

    SimpleHostMem a_host_buf_to_device(sizeof(ADataType) * f_matrix_space_size(m, k, lda, ALayout{}));
    SimpleHostMem b_host_buf_to_device(sizeof(BDataType) * f_matrix_space_size(k, n, ldb, BLayout{}));
    SimpleHostMem c_host_buf_from_device(sizeof(CDataType) * f_matrix_space_size(m, n, ldc, CLayout{}));

    mem_transfer<ADataType, float, SimpleHostMem, SimpleHostMem>(a_host_buf_to_device, a_host_buf, m, k, 1);
    mem_transfer<BDataType, float, SimpleHostMem, SimpleHostMem>(b_host_buf_to_device, b_host_buf, k, n, B_PACKED_K);

    GPU_CHECK_ERROR(hipMemcpy(a_device_buf.GetBuffer(), a_host_buf_to_device.GetBuffer(), m * k * sizeof(ADataType), hipMemcpyHostToDevice));
    GPU_CHECK_ERROR(hipMemcpy(b_device_buf.GetBuffer(), b_host_buf_to_device.GetBuffer(), n * k * sizeof(BDataType), hipMemcpyHostToDevice));
    GPU_CHECK_ERROR(hipMemcpy(scale_device_buf.GetBuffer(), scale_host_buf.GetBuffer(), n * 1 * sizeof(ScaleDataType), hipMemcpyHostToDevice));

    int total_loop=10;
    int warm_ups = 10;
    int i;

// TODO: move this section to a header file

#ifdef ASM_PRINT
    //debug pointer
    float *host_print, *print;
    host_print = (float*)malloc(1024*8);
    GPU_CHECK_ERROR(hipMalloc(&print, 1024*8));
#endif
    struct __attribute__((packed)) {
        void*  ptr_c;
        void*  ptr_a;
        void*  ptr_b;
        void*  ptr_scale;
        unsigned int m;
        unsigned int n;
        unsigned int k;
        unsigned int lda;
        unsigned int ldb;
        unsigned int ldc;
        unsigned int k_per_cta;
        #ifdef ASM_PRINT
        void*  print;
        #endif
    } args;
    size_t arg_size = sizeof(args);
    args.ptr_c  = sk_blocks == 1 ? c_device_buf.GetBuffer() : c_workspace_device_buf.GetBuffer();
    args.ptr_a  = a_device_buf.GetBuffer();
    args.ptr_b  = b_device_buf.GetBuffer();
    args.ptr_scale  = scale_device_buf.GetBuffer();
    args.m      = m;
    args.n      = n;
    args.k      = k;
    args.lda    = lda;
    // args.ldb    = ldb_packed;
    args.ldc    = ldc;
    // args.k_per_cta = k_per_cta;
    #ifdef ASM_PRINT
    args.print  = (void*)print;
    #endif
    CDataType* workspace_ptr = reinterpret_cast<CDataType*>(c_workspace_device_buf.GetBuffer());
    CDataType* c_ptr = reinterpret_cast<CDataType*>(c_device_buf.GetBuffer());
 
    hipStream_t c_stream;
    GPU_CHECK_ERROR(hipStreamCreate(&c_stream));

    for(auto &ker : k_list) {
        std::string kernel_name = ker.kernel_name;
        std::string hsaco_name = ker.kernel_name + ".hsaco";
        GPU_CHECK_ERROR(hipModuleLoad(&module, hsaco_name.c_str()));
        GPU_CHECK_ERROR(hipModuleGetFunction(&kernel_func, module, kernel_name.c_str()));

        int bdx = ker.wg_size;
        int gdx = (m + ker.wg_tile_m - 1) / ker.wg_tile_m; 
        int gdy = (n + ker.wg_tile_n - 1) / ker.wg_tile_n;

        int gdz = sk_blocks;

        printf("grid=[%d, %d, %d], block=[%d]\n", gdx, gdy, gdz, bdx);

        int k_per_cta = ((k + sk_blocks - 1) / sk_blocks + ker.wg_tile_k - 1) / ker.wg_tile_k * ker.wg_tile_k;
        args.k_per_cta = k_per_cta;
        int ldb_packed = n * ker.b_packed_k;
        args.ldb    = ldb_packed;
        void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE,
            &arg_size, HIP_LAUNCH_PARAM_END};
   
        for(i = 0; i < warm_ups; i++){
            GPU_CHECK_ERROR(hipModuleLaunchKernel(kernel_func, gdx,gdy,gdz, bdx,1,1,  0, c_stream, NULL, (void**)&config ));
            if (sk_blocks > 1)     
                tensor_reduce(workspace_ptr, c_ptr, sk_blocks, m * n, c_stream);
            //std::cout<<"safe here"<<std::endl;
        }

#ifdef ASM_PRINT
        int max_i = ker.wg_size;
        GPU_CHECK_ERROR(hipMemcpy(host_print, print, 8*max_i, hipMemcpyDeviceToHost));
        for(int i = 0; i < max_i; i++){
            // if(((uint32_t*)host_print)[2*i+1]!=0x5c005c00)
            float fp32_val = ((float*)host_print)[2*i+1];
            uint32_t fp32_val_bit = __builtin_bit_cast(uint32_t, fp32_val);
            float bf16_lo = __builtin_bit_cast(float, (fp32_val_bit << 16));
            float bf16_hi = __builtin_bit_cast(float, (fp32_val_bit & 0xffff0000));
            printf("Thread%d, PrintVal:0x%x, %d, %f, [%f, %f]\n",((int*) host_print)[2*i], fp32_val_bit, fp32_val_bit, fp32_val, bf16_lo, bf16_hi);
            //std::cout<<"Thread"<<((int*) host_print)[2*i]<<", PrintVal1:"<<(((float16*)host_print)[4*i+2])<<
            //", PrintVal2:"<<( ( (float16*)host_print )[4*i+3] )<<std::endl;
        }    
#endif

        GPU_CHECK_ERROR(hipEventCreate(&evt_00));
        GPU_CHECK_ERROR(hipEventCreate(&evt_11));
        GPU_CHECK_ERROR(hipDeviceSynchronize());
        GPU_CHECK_ERROR(hipEventRecord(evt_00, c_stream));
        for(i = 0; i < total_loop; i++) {
            GPU_CHECK_ERROR(hipModuleLaunchKernel(kernel_func, gdx,gdy,gdz, bdx,1,1,  0, c_stream, NULL, (void**)&config));
            if (sk_blocks > 1)     
                tensor_reduce(workspace_ptr, c_ptr, sk_blocks, m * n, c_stream);
        }
        float elapsed_ms;
        GPU_CHECK_ERROR(hipEventRecord(evt_11, c_stream));
        GPU_CHECK_ERROR(hipEventSynchronize(evt_11));
        GPU_CHECK_ERROR(hipDeviceSynchronize());
        GPU_CHECK_ERROR(hipEventElapsedTime(&elapsed_ms, evt_00, evt_11));
        GPU_CHECK_ERROR(hipEventDestroy(evt_00));
        GPU_CHECK_ERROR(hipEventDestroy(evt_11));

        float time_per_loop = elapsed_ms / total_loop;
        float tflops = (float)2 * m * n * k / time_per_loop / (1024 * 1024 * 1024);
        float bw_gbs = (float)(2 * (m * k + m * n) + n * k) / time_per_loop / (1024 * 1024);
        printf("m: %d, n: %d, k: %d, time: %.3f ms, tflops: %.3f, bw: %.3f GB/s\n",
            m,
            n,
            k,
            time_per_loop,
            tflops,
            bw_gbs);
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
        
            GPU_CHECK_ERROR(hipMemcpy(c_host_buf_from_device.GetBuffer(), c_device_buf.GetBuffer(), ldc * m * sizeof(CDataType), hipMemcpyDeviceToHost));
            bool res = valid_vector<CDataType>(reinterpret_cast<const float*>(c_host_buf.GetBuffer()), reinterpret_cast<const CDataType*>(c_host_buf_from_device.GetBuffer()),  m * n);
            printf(",%s \n", res ? "valid" : "fail");
        }
    }
    
    return 0;
}
