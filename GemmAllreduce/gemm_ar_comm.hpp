#pragma once

#include <iostream>
#include <string>
#include <stdio.h>
#include <vector>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <rocblas/rocblas.h>

template <typename T>
struct TGemmTypes
{
};

template <>
struct TGemmTypes<half>
{
    static const rocblas_datatype hipTypeI = rocblas_datatype_f16_r;
    using dataTypeI = half;
    static const rocblas_datatype hipTypeO = rocblas_datatype_f16_r;
    using dataTypeO = half;
    static const rocblas_datatype hipTypeS = rocblas_datatype_f32_r; // scale type
    using dataTypeS = float;
};

template <>
struct TGemmTypes<float>
{
    static const rocblas_datatype hipTypeI = rocblas_datatype_f32_r;
    using dataTypeI = float;
    static const rocblas_datatype hipTypeO = rocblas_datatype_f32_r;
    using dataTypeO = float;
    static const rocblas_datatype hipTypeS = rocblas_datatype_f32_r; // scale type
    using dataTypeS = float;
};

template <>
struct TGemmTypes<hip_bfloat16>
{
    static const rocblas_datatype hipTypeI = rocblas_datatype_bf16_r;
    using dataTypeI = hip_bfloat16;
    static const rocblas_datatype hipTypeO = rocblas_datatype_bf16_r;
    using dataTypeO = hip_bfloat16;
    static const rocblas_datatype hipTypeS = rocblas_datatype_f32_r; // scale type
    using dataTypeS = float;
};

template <typename T>
struct TGemm
{
    int m, n, k, ldA, ldB, ldC, rA, rB, rC, cA, cB, cC;
    size_t elemA;
    size_t elemB;
    size_t elemC;
  
    size_t bytesA;
    size_t bytesB;
    size_t bytesC;

    using Types = TGemmTypes<T>;
    typename Types::dataTypeI* A{nullptr};
    typename Types::dataTypeI* B{nullptr};
    typename Types::dataTypeO* C{nullptr};

    bool transA, transB;

    typename Types::dataTypeS alpha;
    typename Types::dataTypeS beta;

    TGemm() {}

    // Row Major
    TGemm(int m_, 
          int n_, 
          int k_,
          typename Types::dataTypeI* A_,
          typename Types::dataTypeI* B_,
          typename Types::dataTypeO* C_,
          bool transA_ = false, 
          bool transB_ = false)
    {
        m = m_;
        n = n_;
        k = k_;
        elemA = m * k;
        elemB = n * k;
        elemC = m * n;
        bytesA = sizeof(T) * elemA;
        bytesB = sizeof(T) * elemB;
        bytesC = sizeof(T) * elemC;
  
        A = A_;
        B = B_;
        C = C_;
  
        transA = transA_;
        transB = transB_;
        ldA = transA ? m : k;
        ldB = transB ? k : n;
        ldC = n;

        alpha = 1.f;
        beta = 0.f;
    }
};

#define ROCBLAS_CHECK(status)                                   \
  {                                                            \
    rocblas_status error = status;                             \
    if (error != rocblas_status_success) {                      \
      std::cerr << "rocBLAS Error: " << error                   \
                << " at: " << __FILE__                         \
                << " " << __LINE__                             \
                << std::endl;                                  \
      exit(EXIT_FAILURE);                                      \
    }                                                          \
  }

template <typename T>
rocblas_status inline rocblasGemmEx(rocblas_handle handle, TGemm<T>& gemm)
{
   rocblas_operation opA = gemm.transA ? rocblas_operation_transpose : rocblas_operation_none;
   rocblas_operation opB = gemm.transB ? rocblas_operation_transpose : rocblas_operation_none;

   ROCBLAS_CHECK(rocblas_gemm_ex(handle,
                             opB, opA,
                             gemm.n, gemm.m, gemm.k,
                             &gemm.alpha,
                             gemm.B, TGemm<T>::Types::hipTypeI, gemm.ldB,
                             gemm.A, TGemm<T>::Types::hipTypeI, gemm.ldA,
                             &gemm.beta,
                             gemm.C, TGemm<T>::Types::hipTypeO, gemm.n,
                             gemm.C, TGemm<T>::Types::hipTypeO, gemm.n,
                             rocblas_datatype_f32_r,
                             rocblas_gemm_algo_standard, 0, 0));
   return rocblas_status_success;
}

#define NUM_ITERATIONS 1

template <typename T>
void call_rocBLAS(TGemm<T>& gemm, T* h_C_rocblas) {
  // std::cout << "\nRunning with rocBLAS " << (std::is_same<T, half>::value ? "FP16..." : "FP32...") << std::endl;
  if(std::is_same<T, half>::value){
    std::cout << "\nRunning with rocBLAS FP16..." << std::endl;
  }else if(std::is_same<T, float>::value){
    std::cout << "\nRunning with rocBLAS FP32..." << std::endl;
  }else if(std::is_same<T, hip_bfloat16>::value){
    std::cout << "\nRunning with rocBLAS BF16..." << std::endl;
  }else{
    std::cout << "\nRunning with rocBLAS" << std::endl;
    std::cout << "\nnot support type..." << std::endl;
  }
  
  check_cuda_error(hipMemset(gemm.C, 0, gemm.elemC * sizeof(T)));

  // float* d_zero;
  // int zero_size = 256 * 1024 * 1024;
  // check_cuda_error(hipMalloc((void**)&d_zero,  zero_size));

  rocblas_handle handle;
  ROCBLAS_CHECK(rocblas_create_handle(&handle));
  ROCBLAS_CHECK(rocblasGemmEx(handle, gemm));

  hipEvent_t start, stop;
  check_cuda_error(hipEventCreate(&start));
  check_cuda_error(hipEventCreate(&stop));

    // warm up
  for (int i = 0; i < 1; ++i) {
    ROCBLAS_CHECK(rocblasGemmEx(handle, gemm));
  }

  float total_time = 0.0f;
  check_cuda_error(hipEventRecord(start));

  for (int i = 0; i < NUM_ITERATIONS; ++i) {
    // check_cuda_error(hipMemset(d_zero, 0, zero_size));
    
    ROCBLAS_CHECK(rocblasGemmEx(handle, gemm));
   
  }
  check_cuda_error(hipEventRecord(stop));
  check_cuda_error(hipEventSynchronize(stop));
  check_cuda_error(hipEventElapsedTime(&total_time, start, stop));

  printf("rocBLAS time:  %3.4f ms \n", total_time / NUM_ITERATIONS);

  check_cuda_error(hipMemcpy(h_C_rocblas, gemm.C, gemm.elemC * sizeof(T), hipMemcpyDeviceToHost));
  check_cuda_error(hipDeviceSynchronize());

  check_cuda_error(hipEventDestroy(start));
  check_cuda_error(hipEventDestroy(stop));
  ROCBLAS_CHECK(rocblas_destroy_handle(handle));
}


typedef struct {
    size_t m;
    size_t n;
    size_t k;
    size_t tp;
    size_t dt;
} test_args_t;



