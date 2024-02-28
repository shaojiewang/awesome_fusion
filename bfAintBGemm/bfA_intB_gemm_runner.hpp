#pragma once

#include <hip/hip_runtime.h>
#include <cfloat>

#include "kernel_list.hpp"

struct __attribute__((packed)) kargs{
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
    void*  ptr_workspace; // also use this one to be debug pointer
};

class bfAintBGemmRunner {
public:
    bfAintBGemmRunner(const std::vector<kernel_tunable>& k_vec_,
                      const std::string& hsaco_path,
                      void* ptr_c_,
                      void* ptr_a_,
                      void* ptr_b_,
                      void* ptr_scale_,
                      uint32_t& m_,
                      uint32_t& n_,
                      uint32_t& k_,
                      uint32_t& lda_,
                      uint32_t& ldb_,
                      uint32_t& ldc_,
                      uint32_t& k_per_cta_,
                      void* ptr_workspace_,
                      uint32_t& max_sk_blocks_)
    {
        k_ptr = k_vec_.data();
        args.ptr_c = ptr_c_;
        args.ptr_a = ptr_a_;
        args.ptr_b = ptr_b_;
        args.ptr_scale = ptr_scale_;
        args.m = m_;
        args.n = n_;
        args.k = k_;
        args.lda = lda_;
        args.ldb = ldb_;
        args.ldc = ldc_;
        args.k_per_cta = k_per_cta_;
        args.ptr_workspace = ptr_workspace_;

        k_ptr_len = k_vec_.size();
        max_sk_blocks = max_sk_blocks_;

        for(auto ker : k_vec_)
        {
            hipFunction_t kernel_func;
            std::string kernel_name = ker.kernel_name;
            std::string hsaco_name = hsaco_path + "/" + ker.kernel_name + ".hsaco";
            GPU_CHECK_ERROR(hipModuleLoad(&module, hsaco_name.c_str()));
            GPU_CHECK_ERROR(hipModuleGetFunction(&kernel_func, module, kernel_name.c_str()));
            kernel_func_vec.push_back(kernel_func);
        }
    }
 
    void run(const kernel_tunable& ker,
             hipFunction_t& kernel_func,
             hipStream_t c_stream,
             int sk_blocks)
    {
        size_t arg_size = sizeof(args);
        
        int bdx = ker.wg_size;
        int gdx = (args.m + ker.wg_tile_m - 1) / ker.wg_tile_m; 
        int gdy = (args.n + ker.wg_tile_n - 1) / ker.wg_tile_n;

        int gdz = sk_blocks;
        bfloat16* c_ptr = reinterpret_cast<bfloat16*>(args.ptr_c);
        bfloat16* ptr_workspace = reinterpret_cast<bfloat16*>(args.ptr_workspace);

        // printf("grid=[%d, %d, %d], block=[%d]\n", gdx, gdy, gdz, bdx);
        if (sk_blocks > 1)
        {
            args.ptr_c = args.ptr_workspace;
        }

        int k_per_cta = ((args.k + sk_blocks - 1) / sk_blocks + ker.wg_tile_k - 1) / ker.wg_tile_k * ker.wg_tile_k;
        args.k_per_cta = k_per_cta;
        int ldb_packed = args.n * ker.b_packed_k;
        args.ldb    = ldb_packed;
        void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE,
            &arg_size, HIP_LAUNCH_PARAM_END};
   
        GPU_CHECK_ERROR(hipModuleLaunchKernel(kernel_func, gdx,gdy,gdz, bdx,1,1,  0, c_stream, NULL, (void**)&config ));
        if (sk_blocks > 1)     
            tensor_reduce(ptr_workspace, c_ptr, sk_blocks, args.m * args.n, c_stream);
        // std::cout<<"safe here"<<std::endl;
    }

    auto tune(hipStream_t c_stream,
              int warm_ups,
              int total_loops)
    {
        float min_time = FLT_MAX; 
        int kernel_idx = -1;
        int best_sk_blocks = 0;

        hipEvent_t evt_00, evt_11;

        for(int i = 0; i < k_ptr_len; i++)
        {
            for(int k = 1; k <= max_sk_blocks; k *= 2)
            {
                if(!is_support(k_ptr[i], k))
                {
                    break;
                }
                // warm_up
                for(int n = 0; n < warm_ups; n++)
                {
                    run(k_ptr[i], kernel_func_vec[i], c_stream, k);
                }

                GPU_CHECK_ERROR(hipEventCreate(&evt_00));
                GPU_CHECK_ERROR(hipEventCreate(&evt_11));
                GPU_CHECK_ERROR(hipDeviceSynchronize());
                GPU_CHECK_ERROR(hipEventRecord(evt_00, c_stream));

                // loop
                for(int l = 0; l < total_loops; l++)
                {
                    run(k_ptr[i], kernel_func_vec[i], c_stream, k);               
                }

                float elapsed_ms;
                GPU_CHECK_ERROR(hipEventRecord(evt_11, c_stream));
                GPU_CHECK_ERROR(hipEventSynchronize(evt_11));
                GPU_CHECK_ERROR(hipDeviceSynchronize());
                GPU_CHECK_ERROR(hipEventElapsedTime(&elapsed_ms, evt_00, evt_11));
                GPU_CHECK_ERROR(hipEventDestroy(evt_00));
                GPU_CHECK_ERROR(hipEventDestroy(evt_11));

                if(elapsed_ms < min_time)
                {
                    kernel_idx = i;
                    best_sk_blocks = k;
                    min_time = elapsed_ms;
                }
            }
        }
        return std::make_tuple(kernel_idx, best_sk_blocks);
    }

    size_t get_workspace_size(int sk_blocks)
    {
        if(sk_blocks > 1) 
        {
            return args.m * args.n * sk_blocks;
        }
        else
        {
            return 0;
        }
    }

    bool is_support(const kernel_tunable& ker,
                    int sk_blocks)
    {
        int n = args.n;
        int k = args.k;
        
        if(n % ker.wg_tile_n)
        {
            return false;
        }

        if(k % (ker.wg_tile_k * sk_blocks))
        {
            return false;
        }

        return true;

    }

    void set_workspace_ptr(void* workspace)
    {
        args.ptr_workspace = workspace;
    }
    
    ~bfAintBGemmRunner() = default;    

    const kernel_tunable* k_ptr;
    int k_ptr_len;
    int max_sk_blocks;
    kargs args;
    hipModule_t module;
    std::vector<hipFunction_t> kernel_func_vec;

};

