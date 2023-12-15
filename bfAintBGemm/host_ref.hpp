#pragma once

static inline void gemm_rrr(
    float*  ptr_c,
    const float*  __restrict__ ptr_a,
    const float*  __restrict__ ptr_b,
    std::size_t m,
    std::size_t n,
    std::size_t k,
    std::size_t lda,
    std::size_t ldb,
    std::size_t ldc)
{
    for(auto i_m = 0 ; i_m < m; i_m++) {
        for(auto i_n = 0; i_n < n; i_n++) {
            float acc = 0;
            for(auto i_k = 0; i_k < k; i_k++) {
                acc += ptr_a[i_m * lda + i_k] * ptr_b[i_k * ldb + i_n];
            }
            ptr_c[i_m * ldc + i_n] = acc;
        }
    }
}

