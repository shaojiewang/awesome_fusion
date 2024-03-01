#include "attention_test_common.h""

template<typename T>
bool test_masked_multihead_attention(const test_args_t& test_args)
{
    using Tmha                    = typename mha_type_t<T>::Type;
    const float max_allowed_error = 0.05f;

    int BS = test_args.batch_size;
    int H  = test_args.head_num;
    int L  = test_args.max_seq_len;
    int Dh = test_args.size_per_head;
    int R  = test_args.rotary_dimension;

    GPUBuf<T> q_T(BS * Dh * H), q_bias_T(Dh * H);
    GPUBuf<T> k_T(BS * Dh * H), k_bias_T(Dh * H);
    GPUBuf<T> v_T(BS * Dh * H), v_bias_T(Dh * H);
    GPUBuf<T> kcache_T(BS * Dh * H * L);  // read as [BS, H, Dh/x, L, x]
    GPUBuf<T> vcache_T(BS * Dh * H * L);
    GPUBuf<T> out_T(BS * Dh * H);

    GPUBuf<int> seq_lengths(BS);
    seq_lengths.set((std::vector<int>(BS, L * 4 / 4)).data());

    GPUBuf<float> q_fp32(q_T), q_bias_fp32(q_bias_T);
    GPUBuf<float> k_fp32(k_T), k_bias_fp32(k_bias_T);
    GPUBuf<float> v_fp32(v_T), v_bias_fp32(v_bias_T);
    GPUBuf<float> kcache_fp32(reshape_key_cache(kcache_T, BS, H, Dh, L, 16 / sizeof(T), 16 / sizeof(float)));
    GPUBuf<float> vcache_fp32(vcache_T);
    GPUBuf<float> out_fp32(BS * Dh * H);

    Masked_multihead_attention_params<float> params_fp32;
    set_params_struct(params_fp32,
                      out_fp32.ptr,
                      q_fp32.ptr,
                      q_bias_fp32.ptr,
                      k_fp32.ptr,
                      k_bias_fp32.ptr,
                      v_fp32.ptr,
                      v_bias_fp32.ptr,
                      kcache_fp32.ptr,
                      vcache_fp32.ptr,
                      nullptr,
                      0,
                      BS,
                      1,
                      L,
                      H,
                      Dh,
                      R,
                      L - 1,
                      1.0 / sqrtf(Dh),
                      seq_lengths.ptr,
                      L / 4,
                      (const float*)nullptr,
                      0,
                      0,
                      (int*)nullptr);
    masked_multihead_attention(params_fp32, 0);

    auto mha_ref = out_fp32.to_host_vec();

    Masked_multihead_attention_params<Tmha> params_T;
    set_params_struct(params_T,
                      (Tmha*)out_T.ptr,
                      (Tmha*)q_T.ptr,
                      (Tmha*)q_bias_T.ptr,
                      (Tmha*)k_T.ptr,
                      (Tmha*)k_bias_T.ptr,
                      (Tmha*)v_T.ptr,
                      (Tmha*)v_bias_T.ptr,
                      (Tmha*)kcache_T.ptr,
                      (Tmha*)vcache_T.ptr,
                      nullptr,
                      0,
                      BS,
                      1,
                      L,
                      H,
                      Dh,
                      R,
                      L - 1,
                      1.0 / sqrtf(Dh),
                      seq_lengths.ptr,
                      L / 4,
                      (const Tmha*)nullptr,
                      0,
                      0,
                      (int*)nullptr);
    masked_multihead_attention(params_T, 0);

    auto mha_T_test = GPUBuf<float>(out_T).to_host_vec();

    /* for (int bs = 0; bs < BS; bs++) { */
    /*     for (int h = 0; h < H; h++) { */
    /*         for (int d = 0; d < Dh; d++) { */
    /*             float ref = mha_ref[bs * H * Dh + h * Dh + d]; */
    /*             float test = mha_T_test[bs * H * Dh + h * Dh + d]; */
    /*             const float diff = abs_diff<float>()(ref, test); */
    /*             const float rel_diff = rel_abs_diff<float>()(ref, test); */

    /*             printf("[%d, %d, %d] Error %.2e (=%.2f%%)\n", bs, h, d, diff, rel_diff * 100); */
    /*         } */
    /*     } */
    /* } */
#if 0
    for (int bs = 0; bs < BS; bs++) {
        for (int h = 0; h < H; h++) { 
            for (int d = 0; d < Dh; d++) { 
                float ref = mha_ref[bs * H * Dh + h * Dh + d]; 
                float test = mha_T_test[bs * H * Dh + h * Dh + d]; 
                const float diff = abs_diff<float>()(ref, test); 
                const float rel_diff = rel_abs_diff<float>()(ref, test); 

                printf("[%d, %d, %d] ref=[%f], test=[%f], Error %.2e (=%.2f%%)\n", bs, h, d, ref, test, diff, rel_diff * 100); 
            } 
        } 
    }
#endif 
    std::transform(mha_ref.begin(), mha_ref.end(), mha_T_test.begin(), mha_T_test.begin(), abs_diff<float>());
    const float T_error = *std::max_element(mha_T_test.begin(), mha_T_test.end());

    bool error = false;
    if (T_error > max_allowed_error) {
        error = true;
        printf("Max abs diff = %.2f\n", T_error);
    }

    hipStream_t stream;
    hipStreamCreate(&stream);

    float ms = 0.0f;
    printf("[FP32] ");
    TIMEIT(true, 10, ms, stream, masked_multihead_attention, params_fp32, stream);
    printf("[%s] ", string_rep_t<T>::value.c_str());
    TIMEIT(true, 10, ms, stream, masked_multihead_attention, params_T, stream);

    return !error;
}

int main(int argc, char** argv)
{
    if (argc != 6) {
        printf("[ERROR] Usage: %s batch_size head_num max_seq_len"
               "size_per_head rotary_dim\n",
               argv[0]);
        printf("e.g., %s 32 16 40 256 32\n", argv[0]);
        return EXIT_FAILURE;
    }

    const test_args_t test_args{atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5])};

    struct hipDeviceProp_t prop;
    check_cuda_error(hipGetDeviceProperties(&prop, 0));
    printf("Using device %s\n", prop.name);

    bool global_test_pass = true;
    bool test_pass        = true;

    test_pass = test_masked_multihead_attention<half>(test_args);
    printf("%s", test_pass ? "." : "X");
    global_test_pass |= test_pass;

}


