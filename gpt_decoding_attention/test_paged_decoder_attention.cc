#include "attention_test_common.h"

template<typename T>
float test_paged_masked_multihead_attention(const test_args_t& test_args)
{
    using Tmha                    = typename mha_type_t<T>::Type;
    const float max_allowed_error = 0.05f;

    int BS = test_args.batch_size;
    int H  = test_args.head_num;
    int L  = test_args.max_seq_len;
    int Dh = test_args.size_per_head;
    int R  = test_args.rotary_dimension;
    int PB = test_args.paged_block_size;
    int heads_per_gqa_group = 1;

    GPUBuf<T> q_T(BS * Dh * H), q_bias_T(Dh * H);
    GPUBuf<T> k_T(BS * Dh * H), k_bias_T(Dh * H);
    GPUBuf<T> v_T(BS * Dh * H), v_bias_T(Dh * H);
    GPUBuf<T> kcache_T_transpose(BS * Dh * H * L);  // read as [BS, H, Dh/x, L, x]
    GPUBuf<T> vcache_T_transpose(BS * Dh * H * L);
    GPUBuf<T> kcache_T(BS * Dh * H * L);  // read as [BS, H, Dh/x, L, x]
    GPUBuf<T> vcache_T(BS * Dh * H * L);
    GPUBuf<T> kv_blocks(4 * BS * Dh * H * L);
    GPUBuf<T> out_T(BS * Dh * H);

    size_t num_blocks_per_bs = (L + PB - 1) / PB;
    size_t num_blocks = num_blocks_per_bs * BS;

    GPUBuf<size_t> k_block_offset(num_blocks);
    GPUBuf<size_t> k_batch_offset(BS);
    GPUBuf<size_t> v_block_offset(num_blocks);
    GPUBuf<size_t> v_batch_offset(BS);

    invokeSetPageBlockOffset(k_block_offset.ptr, PB * Dh * H * 2, 0, num_blocks);
    invokeSetPageBlockOffset(v_block_offset.ptr, PB * Dh * H * 2, 2 * num_blocks * PB * Dh * H, num_blocks);
    check_cuda_error(hipDeviceSynchronize());
    invokeSetBatchBlockPtrs(k_batch_offset.ptr, k_block_offset.ptr, BS, num_blocks_per_bs);
    invokeSetBatchBlockPtrs(v_batch_offset.ptr, v_block_offset.ptr, BS, num_blocks_per_bs);
    check_cuda_error(hipDeviceSynchronize());

    GPUBuf<int> seq_lengths(BS);
    seq_lengths.set((std::vector<int>(BS, L * 4 / 4)).data());

#if 1
    invokeTranspose4dBatchMajor(
        kcache_T_transpose.ptr,
        vcache_T_transpose.ptr,
        kcache_T.ptr,
        vcache_T.ptr,
        BS,
        L - 1,
        L,
        Dh,
        H,
        (hipStream_t)(0)
    );
    invokeTranspose4dBatchMajorWithKVCachePtr(
        reinterpret_cast<T*>(kv_blocks.ptr),
        reinterpret_cast<size_t**>(k_batch_offset.ptr),
        reinterpret_cast<size_t**>(v_batch_offset.ptr),
        reinterpret_cast<T*>(kcache_T.ptr),
        reinterpret_cast<T*>(vcache_T.ptr),
        reinterpret_cast<int*>(seq_lengths.ptr),
        PB,
        0,
        BS,
        L - 1,
        L,
        Dh,
        H,
        (hipStream_t)(0)
    );
#endif        
    check_cuda_error(hipDeviceSynchronize());
    
    GPUBuf<int> cur_timesteps(BS);
    cur_timesteps.set((std::vector<int>(BS, L)).data());

    GPUBuf<float> q_fp32(q_T), q_bias_fp32(q_bias_T);
    GPUBuf<float> k_fp32(k_T), k_bias_fp32(k_bias_T);
    GPUBuf<float> v_fp32(v_T), v_bias_fp32(v_bias_T);
    GPUBuf<float> kcache_fp32(reshape_key_cache(kcache_T_transpose, BS, H, Dh, L, 16 / sizeof(T), 16 / sizeof(float)));
    GPUBuf<float> vcache_fp32(vcache_T_transpose);
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
                      heads_per_gqa_group, 
                      Dh,
                      R,
                      L - 1,
                      1.0, // / sqrtf(Dh),
                      seq_lengths.ptr,
                      L / 4,
                      (const float*)nullptr,
                      0,
                      PB,
                      (int*)nullptr);
    masked_multihead_attention(params_fp32, 0);
    check_cuda_error(hipDeviceSynchronize());

    GPUBuf<T> out_fp32_bf16(out_fp32);
    GPUBuf<float> out_fp32_bf16_fp32(out_fp32_bf16);
    auto mha_ref = out_fp32_bf16_fp32.to_host_vec();

    Paged_masked_multihead_attention_params<Tmha> params_T;
    set_params_struct(params_T,
                      (Tmha*)out_T.ptr,
                      (Tmha*)q_T.ptr,
                      (Tmha*)q_bias_T.ptr,
                      (Tmha*)k_T.ptr,
                      (Tmha*)k_bias_T.ptr,
                      (Tmha*)v_T.ptr,
                      (Tmha*)v_bias_T.ptr,
                      (Tmha*)kv_blocks.ptr,
                      (size_t**)k_batch_offset.ptr,
                      (size_t**)v_batch_offset.ptr,
                      nullptr,
                      0,
                      BS,
                      1,
                      L,
                      H,
                      heads_per_gqa_group,
                      Dh,
                      R,
                      L - 1,
                      1.0, // / sqrtf(Dh),
                      seq_lengths.ptr,
                      L,
                      (const Tmha*)nullptr,
                      0,
                      PB,
                      (int*)cur_timesteps.ptr);
    printf("params.memory_max_len=%d\n", params_T.memory_max_len);
    paged_masked_multihead_attention(params_T, 0);
    check_cuda_error(hipDeviceSynchronize());


    
    auto mha_T_test = GPUBuf<float>(out_T).to_host_vec();

#if VIEW_ERR_POINT
    for (int bs = 0; bs < BS; bs++) {
        for (int h = 0; h < H; h++) { 
            for (int d = 0; d < Dh; d++) { 
                float ref = mha_ref[bs * H * Dh + h * Dh + d]; 
                float test = mha_T_test[bs * H * Dh + h * Dh + d]; 
                const float diff = abs_diff<float>()(ref, test); 
                const float rel_diff = rel_abs_diff<float>()(ref, test); 
                if(rel_diff > 0.01)
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
    check_cuda_error(hipStreamCreate(&stream));

    float ms = 0.0f;
    printf("[FP32] ");
    TIMEIT(true, 10, ms, stream, masked_multihead_attention, params_fp32, stream);
    printf("[%s] ", string_rep_t<T>::value.c_str());
    TIMEIT(true, 10, ms, stream, paged_masked_multihead_attention, params_T, stream);

    printf("%s\n", !error ? "." : "X");
    return ms;
}

int main(int argc, char** argv)
{
    if (argc != 8) {
        printf("[ERROR] Usage: %s batch_size head_num max_seq_len"
               "max_output_len size_per_head rotary_dim page_block_size\n",
               argv[0]);
        printf("e.g., %s 32 16 2048 240 128 128 64\n", argv[0]);
        return EXIT_FAILURE;
    }

    test_args_t test_args{atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), atoi(argv[6]), atoi(argv[7])};
    // assert(test_args.paged_block_size == 8);

    struct hipDeviceProp_t prop;
    check_cuda_error(hipGetDeviceProperties(&prop, 0));
    printf("Using device %s\n", prop.name);

    float total_time_fp16 = 0.0f, total_time_bf16 = 0.0f;
    for(int i = 0; i < test_args.max_output_len - 1; i++){
        total_time_fp16 += test_paged_masked_multihead_attention<half>(test_args);
        total_time_bf16 += test_paged_masked_multihead_attention<__nv_bfloat16>(test_args);
        test_args.max_seq_len ++;
    }

    printf("fp16 avg time = %f\n", total_time_fp16 / (test_args.max_output_len - 1));
    printf("bf16 avg time = %f\n", total_time_bf16 / (test_args.max_output_len - 1));
    
}


