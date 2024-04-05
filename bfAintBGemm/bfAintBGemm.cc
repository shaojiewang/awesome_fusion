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
#include "bfA_intB_gemm_runner.hpp"

#include "kernel_list.hpp"

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

int main(int argc, char ** argv)
{
    int validation = 0;
    int init_method = 0;

    uint32_t m = 32;
    uint32_t n = 64 * 2;
    uint32_t k = 256 * 2;

    std::string hsaco_path;
    if(argc >= 2)
    {
        hsaco_path = argv[1];
    }
    if(argc >= 4) 
    {
        validation = atoi(argv[2]);
        init_method = atoi(argv[3]);
    }
    if(argc >= 7) {
        m = atoi(argv[4]);
        n = atoi(argv[5]);
        k = atoi(argv[6]);
    }

    uint32_t lda = k;
    uint32_t ldb = n;
    uint32_t ldc = n;

    if(argc >= 10) 
    {
        lda = atoi(argv[7]);
        ldb = atoi(argv[8]);
        ldc = atoi(argv[9]);
    }

    // get kernel list
    std::vector<kernel_tunable> k_list = get_kernel_list();

    // hipModule_t module;
    // hipFunction_t kernel_func;
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

    uint32_t max_sk_blocks = 16;
    
    SimpleDeviceMem a_device_buf(sizeof(ADataType) * f_matrix_space_size(m, k, lda, ALayout{}));
    SimpleDeviceMem b_device_buf(sizeof(BDataType) * f_matrix_space_size(k, n, ldb, BLayout{}));
    SimpleDeviceMem c_device_buf(sizeof(CDataType) * f_matrix_space_size(m, n, ldc, CLayout{}));
    SimpleDeviceMem c_workspace_device_buf(sizeof(CDataType) * f_matrix_space_size(m * max_sk_blocks, n, ldc, CLayout{}));
    SimpleDeviceMem scale_device_buf(sizeof(ScaleDataType) * f_matrix_space_size(n, 1, 1, ScaleLayout{}));
    
    SimpleHostMem a_host_buf(sizeof(float) * f_matrix_space_size(m, k, lda, ALayout{}));
    SimpleHostMem b_host_buf(sizeof(float) * f_matrix_space_size(k, n, ldb, BLayout{}));
    SimpleHostMem c_host_buf(sizeof(float) * f_matrix_space_size(m, n, ldc, CLayout{}));
    SimpleHostMem scale_host_buf(sizeof(float) * f_matrix_space_size(n, 1, 1, ScaleLayout{}));

    if(validation)
    {
        if(init_method == 1)
        {
            rand_vector_2d_int_a(reinterpret_cast<float*>(a_host_buf.GetBuffer()), m, k, lda);
            rand_vector_2d_int_b(reinterpret_cast<float*>(b_host_buf.GetBuffer()), k, n, ldb);
            rand_vector_2d_int_scale(reinterpret_cast<float*>(scale_host_buf.GetBuffer()), n, 1, 1);
        }
        else if(init_method > 1)
        {
            rand_vector_2d(reinterpret_cast<float*>(a_host_buf.GetBuffer()), m, k, lda);
            rand_vector_2d_int_b(reinterpret_cast<float*>(b_host_buf.GetBuffer()), k, n, ldb);
            rand_vector_2d(reinterpret_cast<float*>(scale_host_buf.GetBuffer()), n, 1, 1);
        }
    }

    SimpleHostMem a_host_buf_to_device(sizeof(ADataType) * f_matrix_space_size(m, k, lda, ALayout{}));
    SimpleHostMem b_host_buf_to_device(sizeof(BDataType) * f_matrix_space_size(k, n, ldb, BLayout{}));
    SimpleHostMem c_host_buf_from_device(sizeof(CDataType) * f_matrix_space_size(m, n, ldc, CLayout{}));

    mem_transfer<ADataType, float, SimpleHostMem, SimpleHostMem>(a_host_buf_to_device, a_host_buf, m, k, 1);
    mem_transfer<float, ADataType, SimpleHostMem, SimpleHostMem>(a_host_buf, a_host_buf_to_device, m, k, 1);
    mem_transfer<BDataType, float, SimpleHostMem, SimpleHostMem>(b_host_buf_to_device, b_host_buf, k, n, B_PACKED_K);

    GPU_CHECK_ERROR(hipMemcpy(a_device_buf.GetBuffer(), a_host_buf_to_device.GetBuffer(), m * k * sizeof(ADataType), hipMemcpyHostToDevice));
    GPU_CHECK_ERROR(hipMemcpy(b_device_buf.GetBuffer(), b_host_buf_to_device.GetBuffer(), n * k * sizeof(BDataType), hipMemcpyHostToDevice));
    GPU_CHECK_ERROR(hipMemcpy(scale_device_buf.GetBuffer(), scale_host_buf.GetBuffer(), n * 1 * sizeof(ScaleDataType), hipMemcpyHostToDevice));

    int total_loop = 10;
    int warm_ups = 10;

// TODO: move this section to a header file

#ifdef ASM_PRINT
    //debug pointer
    float *host_print, *print;
    uint32_t print_sk_blocks = 1;
    host_print = (float*)malloc(1024*8);
    GPU_CHECK_ERROR(hipMalloc(&print, 1024*8));
#endif

    bfAintBGemmRunner bfa_intb_gemm_runner(k_list,
                                           hsaco_path,  
                                           c_device_buf.GetBuffer(),
                                           a_device_buf.GetBuffer(),
                                           b_device_buf.GetBuffer(),
                                           scale_device_buf.GetBuffer(),
                                           m,
                                           n,
                                           k,
                                           lda,
                                           ldb,
                                           ldc,
                                           k,
#ifdef ASM_PRINT
                                           print,
                                           print_sk_blocks
#else
                                           nullptr,
                                           max_sk_blocks
#endif
                                           );

 
    hipStream_t c_stream;
    GPU_CHECK_ERROR(hipStreamCreate(&c_stream));

    if(max_sk_blocks > 1)
        bfa_intb_gemm_runner.set_workspace_ptr(c_workspace_device_buf.GetBuffer());
    auto [sol_idx, sk_blocks] = bfa_intb_gemm_runner.tune(c_stream, warm_ups, total_loop);

    float elapsed_ms;
    GPU_CHECK_ERROR(hipEventCreate(&evt_00));
    GPU_CHECK_ERROR(hipEventCreate(&evt_11));
    GPU_CHECK_ERROR(hipDeviceSynchronize());
    GPU_CHECK_ERROR(hipEventRecord(evt_00, c_stream));

    for(int i = 0; i < total_loop; i++)
        bfa_intb_gemm_runner.run(bfa_intb_gemm_runner.k_ptr[sol_idx], bfa_intb_gemm_runner.kernel_func_vec[sol_idx], c_stream, sk_blocks);

    GPU_CHECK_ERROR(hipEventRecord(evt_11, c_stream));
    GPU_CHECK_ERROR(hipEventSynchronize(evt_11));
    GPU_CHECK_ERROR(hipDeviceSynchronize());
    GPU_CHECK_ERROR(hipEventElapsedTime(&elapsed_ms, evt_00, evt_11));
    GPU_CHECK_ERROR(hipEventDestroy(evt_00));
    GPU_CHECK_ERROR(hipEventDestroy(evt_11));

#ifdef ASM_PRINT
    int max_i = bfa_intb_gemm_runner.k_ptr[sol_idx].wg_size;
    GPU_CHECK_ERROR(hipMemcpy(host_print, print, 8*max_i, hipMemcpyDeviceToHost));
    for(int i = 0; i < max_i; i++){
        // if(((uint32_t*)host_print)[2*i+1]!=0x5c005c00)
        float fp32_val = ((float*)host_print)[2*i+1];
        uint32_t fp32_val_bit = __builtin_bit_cast(uint32_t, fp32_val);
        float bf16_lo = __builtin_bit_cast(float, (fp32_val_bit << 16));
        float bf16_hi = __builtin_bit_cast(float, (fp32_val_bit & 0xffff0000));
        printf("%dth: Thread%d, PrintVal:0x%x, %d, %f, [%f, %f]\n", i, ((int*) host_print)[2*i], fp32_val_bit, fp32_val_bit, fp32_val, bf16_lo, bf16_hi);
        //std::cout<<"Thread"<<((int*) host_print)[2*i]<<", PrintVal1:"<<(((float16*)host_print)[4*i+2])<<
        //", PrintVal2:"<<( ( (float16*)host_print )[4*i+3] )<<std::endl;
        float fp32_a = ((float*)(a_device_buf.GetBuffer()))[i];
        uint32_t fp32_a_val_bit =  __builtin_bit_cast(uint32_t, fp32_a);
        float a_lo = __builtin_bit_cast(float, (fp32_a_val_bit << 16));
        float a_hi = __builtin_bit_cast(float, (fp32_a_val_bit & 0xffff0000));
        printf("%dth: Thread%d, PrintVal:0x%x, %d, %f, [%f, %f]\n", i, ((int*) host_print)[2*i], fp32_a_val_bit, fp32_a_val_bit, fp32_a, a_lo, a_hi);
    }    
#endif


    float time_per_loop = elapsed_ms / total_loop;
    float tflops = (float)2 * m * n * k / time_per_loop / (1024 * 1024 * 1024);
    float bw_gbs = (float)(2 * (m * k + m * n) + n * k) / time_per_loop / (1024 * 1024);
    
    printf("best [sol, sk_blocks]: [%d, %d], m: %d, n: %d, k: %d, time: %.3f ms, tflops: %.3f, bw: %.3f GB/s\n",
        sol_idx, sk_blocks,
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
    
    return 0;
}
