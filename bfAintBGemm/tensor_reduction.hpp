#include "gpu_utils.hpp"
#include "datatype.hpp"

template <typename T>
struct acc_type {
};

template <>
struct acc_type<bfloat16> {
    using Type = float;
};

template <int cta_size,
          typename T>
__global__ void tensor_reduce_kernel(T* in, T* out, int reduce_dim, int remain_dim) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    T* tmp_in = in + bid * cta_size;
    T* tmp_out = out + bid * cta_size;
    using acc_t = typename acc_type<T>::Type;
    acc_t res = 0;
    acc_t in_val = 0;
    for (int i = 0; i < reduce_dim; i++) {
        in_val = type_convert<acc_t, T>(tmp_in[tid]);
        res += in_val;
        tmp_in += remain_dim;
    }
    tmp_out[tid] = type_convert<T, acc_t>(res);
    
}

template <typename T>
void tensor_reduce(T* in, T* out, int reduce_dim, int remain_dim, hipStream_t stream) {
    constexpr int cta_size = 128;
    int gdx = (remain_dim + cta_size - 1) / cta_size;
    dim3 grd(gdx);
    dim3 bld(cta_size);
    tensor_reduce_kernel<cta_size, T><<<grd, bld, 0, stream>>>(in, out, reduce_dim, remain_dim);
}

template 
void tensor_reduce<bfloat16>(bfloat16* in, bfloat16* out, int reduce_dim, int remain_dim, hipStream_t stream);

