#include <hip/hip_runtime_api.h>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/time.h>
#include <vector>
#include "nccl.h"
#include <mpi.h>

#include "hip_type_utils.cuh"
#include "custom_ar_comm.h"
#include "gemm_ar_comm.hpp"
#include "gemm_matrix_layout.hpp"
#include "simple_mem_buf.hpp"
#include "random_gen.hpp"
#include "matrix_transpose.hpp"
#include "matrix_elementwise.hpp"


// whether to use custom kernel[1] or rccl[0]
const int custom_ar = 1;
// num of elements to do all reduce
const int AR_NUM = 8192;

#define TOTAL_NUM 100
#define WARM_UP_NUM 10

#define MAX_WORLD_SIZE 8
#define MAX_HANDLE_NUM 8

using namespace awesome_fusion;

using Row = awesome_fusion::gemm_layout::gemm::RowMajor;
using Col = awesome_fusion::gemm_layout::gemm::ColumnMajor;

using ALayout = Row;
using BLayout = Row;
using ScaleLayout = Row;
using CLayout = Row;

using Half = half;
using BHalf = __nv_bfloat16;

template <class ADataType, 
          class BDataType,
          class ScaleDataType,
          class CDataType,
          class ComputeDataType>
int gemm_ar(const test_args_t& args, const int& rank, const int& world_size)
{
    printf("m, n, k, tp, dt=[%zu %zu %zu %zu %zu]\n",
        args.m,
        args.n,
        args.k,
        args.tp,
        args.dt);
    
    // init shape
    int m = args.m;
    int n = args.n;
    int k = args.k;
    int tp = args.tp;
    int k_per_card = k / tp;

    int lda = k_per_card;
    int ldb = n;
    int ldc = n;

    int lda_ref = k;
    int ldb_ref = n;
    int ldc_ref = n;
    
    // assertion
    if (k % (tp * 64) != 0) return 0;
    if (world_size > MAX_WORLD_SIZE) return 0; 

    // initialize custom all reduce 
    std::vector<std::shared_ptr<AbstractCustomComm>> custom_all_reduce_comms;
    initCustomAllReduceComm<uint16_t>(&custom_all_reduce_comms, custom_ar, world_size);
    
    // set device
    int device, device_count;
    check_cuda_error(hipGetDeviceCount(&device_count));
    check_cuda_error(hipSetDevice(rank % device_count));
    check_cuda_error(hipGetDevice(&device));
    struct hipDeviceProp_t prop;
    check_cuda_error(hipGetDeviceProperties(&prop, device));
    printf("Device %s\n", prop.name);
    printf("P%d is running with GPU #%d.\n", rank, device);
    
    // initialize rccl
    NcclParam tensor_para;
    NcclParam pipeline_para;
    ftNcclInitialize(tensor_para, pipeline_para, world_size, 1);
    
    // malloc tensor
    auto f_matrix_space_size = 
        [](std::size_t nRow, std::size_t nCol, std::size_t stride, auto layout){
            using Layout = decltype(layout);
            if constexpr(std::is_same<Layout, Row>::value) {
                return (nRow - 1) * stride + nCol;
            } else {
                return (nCol - 1) * stride + nRow;
            }
        };

    using DeviceMemCached = SimpleDeviceMem<false>;
    using DeviceMemUncached = SimpleDeviceMem<true>;

    DeviceMemCached a_device_buf_compute(sizeof(ADataType) * f_matrix_space_size(m, k_per_card, lda, ALayout{}));
    DeviceMemCached b_device_buf_compute(sizeof(BDataType) * f_matrix_space_size(k_per_card, n, ldb, BLayout{}));

    DeviceMemCached a_device_buf(sizeof(ADataType) * f_matrix_space_size(m, k_per_card, lda, ALayout{}));
    DeviceMemCached b_device_buf(sizeof(BDataType) * f_matrix_space_size(k_per_card, n, ldb, BLayout{}));
    DeviceMemUncached c_device_buf(sizeof(CDataType) * f_matrix_space_size(m, n, ldc, CLayout{}));
    // SimpleDeviceMem c_workspace_device_buf(sizeof(CDataType) * f_matrix_space_size(m * max_sk_blocks, n, ldc, CLayout{}));
    DeviceMemCached scale_device_buf(sizeof(ScaleDataType) * f_matrix_space_size(n, 1, 1, ScaleLayout{}));
    
    DeviceMemCached a_device_buf_ref(sizeof(ADataType) * f_matrix_space_size(m, k, lda_ref, ALayout{}));
    DeviceMemCached b_device_buf_ref(sizeof(BDataType) * f_matrix_space_size(k, n, ldb_ref, BLayout{}));
    DeviceMemCached b_device_buf_ref_compute(sizeof(ComputeDataType) * f_matrix_space_size(k, n, ldb_ref, BLayout{}));
    DeviceMemCached c_device_buf_ref(sizeof(CDataType) * f_matrix_space_size(m, n, ldc_ref, CLayout{}));
    // SimpleDeviceMem c_workspace_device_buf(sizeof(CDataType) * f_matrix_space_size(m * max_sk_blocks, n, ldc, CLayout{}));
    DeviceMemCached scale_device_buf_ref(sizeof(ScaleDataType) * f_matrix_space_size(n, 1, 1, ScaleLayout{}));
    
    SimpleHostMem a_host_buf(sizeof(float) * f_matrix_space_size(m, k, lda_ref, ALayout{}));
    SimpleHostMem b_host_buf(sizeof(float) * f_matrix_space_size(k, n, ldb_ref, BLayout{}));
    SimpleHostMem c_host_buf(sizeof(float) * f_matrix_space_size(m, n, ldc_ref, CLayout{}));
    SimpleHostMem scale_host_buf(sizeof(float) * f_matrix_space_size(n, 1, 1, ScaleLayout{}));
  

    // pointer communication via ipc
    void* init_a_buf_ptrs[MAX_WORLD_SIZE];
    void* init_a_buf_ref_ptrs[MAX_WORLD_SIZE];
    void* init_b_buf_ptrs[MAX_WORLD_SIZE];
    void* init_b_buf_ref_ptrs[MAX_WORLD_SIZE];
    void* init_scale_buf_ptrs[MAX_WORLD_SIZE];
    void* init_scale_buf_ref_ptrs[MAX_WORLD_SIZE];
    void* out_c_buf_ptrs[MAX_WORLD_SIZE];

    for (int i = 0; i < world_size; i++)
    {
        hipIpcMemHandle_t handle[MAX_HANDLE_NUM];
        if (rank == i)
        {
            init_a_buf_ptrs[i] = reinterpret_cast<void*>(a_device_buf.GetBuffer());
            check_cuda_error(hipIpcGetMemHandle(&(handle[0]), init_a_buf_ptrs[i]));
            init_a_buf_ref_ptrs[i] = reinterpret_cast<void*>(a_device_buf_ref.GetBuffer());
            check_cuda_error(hipIpcGetMemHandle(&(handle[1]), init_a_buf_ref_ptrs[i]));
            init_b_buf_ptrs[i] = reinterpret_cast<void*>(b_device_buf.GetBuffer());
            check_cuda_error(hipIpcGetMemHandle(&(handle[2]), init_b_buf_ptrs[i]));
            init_b_buf_ref_ptrs[i] = reinterpret_cast<void*>(b_device_buf_ref.GetBuffer());
            check_cuda_error(hipIpcGetMemHandle(&(handle[3]), init_b_buf_ref_ptrs[i]));
            init_scale_buf_ptrs[i] = reinterpret_cast<void*>(scale_device_buf.GetBuffer());
            check_cuda_error(hipIpcGetMemHandle(&(handle[4]), init_scale_buf_ptrs[i]));
            init_scale_buf_ref_ptrs[i] = reinterpret_cast<void*>(scale_device_buf_ref.GetBuffer());
            check_cuda_error(hipIpcGetMemHandle(&(handle[5]), init_scale_buf_ref_ptrs[i]));
        }
        MPI_Bcast(&(handle[0]), sizeof(hipIpcMemHandle_t), MPI_CHAR, i, MPI_COMM_WORLD);
        MPI_Bcast(&(handle[1]), sizeof(hipIpcMemHandle_t), MPI_CHAR, i, MPI_COMM_WORLD);
        MPI_Bcast(&(handle[2]), sizeof(hipIpcMemHandle_t), MPI_CHAR, i, MPI_COMM_WORLD);
        MPI_Bcast(&(handle[3]), sizeof(hipIpcMemHandle_t), MPI_CHAR, i, MPI_COMM_WORLD);
        MPI_Bcast(&(handle[4]), sizeof(hipIpcMemHandle_t), MPI_CHAR, i, MPI_COMM_WORLD);
        MPI_Bcast(&(handle[5]), sizeof(hipIpcMemHandle_t), MPI_CHAR, i, MPI_COMM_WORLD);
        if (rank != i)
        {
            check_cuda_error(hipIpcOpenMemHandle((void **)&(init_a_buf_ptrs[i]), handle[0], hipIpcMemLazyEnablePeerAccess));
            check_cuda_error(hipIpcOpenMemHandle((void **)&(init_a_buf_ref_ptrs[i]), handle[1], hipIpcMemLazyEnablePeerAccess));
            check_cuda_error(hipIpcOpenMemHandle((void **)&(init_b_buf_ptrs[i]), handle[2], hipIpcMemLazyEnablePeerAccess));
            check_cuda_error(hipIpcOpenMemHandle((void **)&(init_b_buf_ref_ptrs[i]), handle[3], hipIpcMemLazyEnablePeerAccess));
            check_cuda_error(hipIpcOpenMemHandle((void **)&(init_scale_buf_ptrs[i]), handle[4], hipIpcMemLazyEnablePeerAccess));
            check_cuda_error(hipIpcOpenMemHandle((void **)&(init_scale_buf_ref_ptrs[i]), handle[5], hipIpcMemLazyEnablePeerAccess));
        }
        
    }

    // init tensor on rank 0
    if (rank == 0)
    {
        cudaRandomUniform<ADataType>(reinterpret_cast<ADataType*>(a_device_buf_ref.GetBuffer()), m * k);
        cudaRandomUniform<ComputeDataType>(reinterpret_cast<ComputeDataType*>(b_device_buf_ref.GetBuffer()), n * k);
        cudaRandomUniform<ScaleDataType>(reinterpret_cast<ScaleDataType*>(scale_device_buf_ref.GetBuffer()), n);
        
        for (int i =0; i < world_size; i++)
        {
            check_cuda_error(hipMemcpy(init_a_buf_ptrs[i], (char*)(init_a_buf_ref_ptrs[0]) + i * sizeof(ADataType) * m * k_per_card, sizeof(ADataType) * m * k_per_card, hipMemcpyDeviceToDevice));
            check_cuda_error(hipMemcpy(init_b_buf_ptrs[i], (char*)(init_b_buf_ref_ptrs[0]) + i * sizeof(BDataType) * n * k_per_card, sizeof(BDataType) * n * k_per_card, hipMemcpyDeviceToDevice));
            check_cuda_error(hipMemcpy(init_scale_buf_ptrs[i], (char*)(init_scale_buf_ref_ptrs[0]), sizeof(ScaleDataType) * n, hipMemcpyDeviceToDevice));
            if (i != 0)
            {
                check_cuda_error(hipMemcpy(init_a_buf_ref_ptrs[i], (char*)(init_a_buf_ref_ptrs[0]), sizeof(ADataType) * m * k, hipMemcpyDeviceToDevice));
                check_cuda_error(hipMemcpy(init_b_buf_ref_ptrs[i], (char*)(init_b_buf_ref_ptrs[0]), sizeof(BDataType) * n * k, hipMemcpyDeviceToDevice));
                check_cuda_error(hipMemcpy(init_scale_buf_ref_ptrs[i], (char*)(init_scale_buf_ref_ptrs[0]), sizeof(ScaleDataType) * n, hipMemcpyDeviceToDevice));
            }
        }
    }

    check_cuda_error(hipDeviceSynchronize());

    // fix compute b ref in bfloat16
    invokeMatrixElementwiseScale(reinterpret_cast<hip_bfloat16*>(b_device_buf_ref_compute.GetBuffer()), reinterpret_cast<int8_t*>(b_device_buf_ref.GetBuffer()), reinterpret_cast<float*>(scale_device_buf_ref.GetBuffer()), n, k);

    // reference result by rocblas
    ADataType* d_A = reinterpret_cast<ADataType*>(init_a_buf_ref_ptrs[rank]);
    ComputeDataType* d_B = reinterpret_cast<ComputeDataType*>(b_device_buf_ref_compute.GetBuffer());
    CDataType* d_C = reinterpret_cast<CDataType*>(c_device_buf_ref.GetBuffer());
    TGemm<ADataType> gemm_t(m, n, k, d_A, d_B, d_C, true, false);
    call_rocBLAS(gemm_t, reinterpret_cast<CDataType*>(c_host_buf.GetBuffer()));

    printf("rank %d, c_ref is [0x%x]\n", rank, *(int*)(c_device_buf_ref.GetBuffer()));

    // check broadcast res
    if (rank == 0)
    {
        printf("a buf ref is [0x%x, 0x%x, 0x%x, 0x%x]\n", 
            *(int*)(init_a_buf_ref_ptrs[0]),
            *(int*)((char*)(init_a_buf_ref_ptrs[0]) + sizeof(ADataType) * m * k_per_card),
            *(int*)((char*)(init_a_buf_ref_ptrs[0]) + 2 * sizeof(ADataType) * m * k_per_card),
            *(int*)((char*)(init_a_buf_ref_ptrs[0]) + 3 * sizeof(ADataType) * m * k_per_card));
        printf("b buf ref is [0x%x, 0x%x, 0x%x, 0x%x]\n", 
            *(int*)(init_b_buf_ref_ptrs[0]),
            *(int*)((char*)(init_b_buf_ref_ptrs[0]) + sizeof(ComputeDataType) * n * k_per_card),
            *(int*)((char*)(init_b_buf_ref_ptrs[0]) + 2 * sizeof(ComputeDataType) * n * k_per_card),
            *(int*)((char*)(init_b_buf_ref_ptrs[0]) + 3 * sizeof(ComputeDataType) * n * k_per_card));
        printf("scale buf ref is [0x%x]\n", 
            *(int*)(init_scale_buf_ref_ptrs[0]));
    }
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < world_size; i++)
    {
        if (i == rank)
        {
            printf("in rank [%d], a buf is [0x%x]\n", rank, *(int*)(init_a_buf_ptrs[i]));
            printf("in rank [%d], b buf is [0x%x]\n", rank, *(int*)(init_b_buf_ptrs[i]));
            printf("in rank [%d], scale buf is [0x%x]\n", rank, *(int*)(init_scale_buf_ptrs[i]));
            printf("in rank [%d], a buf ref is [0x%x]\n", rank, *(int*)(init_a_buf_ref_ptrs[i]));
            printf("in rank [%d], b buf ref is [0x%x]\n", rank, *(int*)(init_b_buf_ref_ptrs[i]));
            printf("in rank [%d], scale buf ref is [0x%x]\n", rank, *(int*)(init_scale_buf_ref_ptrs[i]));
        }
    }

    // add matrix transpose code
    // 1. transpose A matrix
    invokeMatrixTranspose(reinterpret_cast<hip_bfloat16*>(a_device_buf_compute.GetBuffer()), reinterpret_cast<hip_bfloat16*>(a_device_buf.GetBuffer()), m, k_per_card, 0);
    // 2. transpose and interleave B matrix
    invokeMatrixBatchedTranspose(reinterpret_cast<hip_bfloat16*>(b_device_buf_compute.GetBuffer()), reinterpret_cast<hip_bfloat16*>(b_device_buf.GetBuffer()), 16 * k, 16, n / 16, 0);

    // output buff
    half *dev_buff, host_buff[AR_NUM], *tmp;
    check_cuda_error(hipMalloc((void**)&tmp, AR_NUM*sizeof(uint16_t)));
    
    if(custom_ar == 1){
        static_cast<CustomAllReduceComm<uint16_t>*>(custom_all_reduce_comms[rank].get())->param_.local_output_buffer_ptr = (uint16_t*)tmp;
        dev_buff = (half*)static_cast<CustomAllReduceComm<uint16_t>*>(custom_all_reduce_comms[rank].get())->param_.peer_comm_buffer_ptrs[rank];
    }
    else{
        dev_buff = tmp;
    }

    // replaced with ops like ffn
    for(int i = 0; i< AR_NUM; i++){
        host_buff[i] = __float2half(1.0);
    }
    check_cuda_error(hipMemcpyHtoD(dev_buff, &host_buff, sizeof(uint16_t)*AR_NUM));
    check_cuda_error(hipDeviceSynchronize()); 
     
    MPI_Barrier(MPI_COMM_WORLD);

    // warm up
    for (int i = 0; i < WARM_UP_NUM; i++)
    {
        if(custom_ar == 1)
            custom_all_reduce_comms[rank]->customAllReduce(AR_NUM, nullptr);
        else
            ftNcclAllReduceSum(dev_buff, dev_buff, AR_NUM, tensor_para, nullptr);
    }

    // perform all reduce
    hipStream_t stream;
    check_cuda_error(hipStreamCreate(&stream));
    hipEvent_t event_s, event_e;
    check_cuda_error(hipEventCreate(&event_s));
    check_cuda_error(hipEventCreate(&event_e));
    check_cuda_error(hipEventRecord(event_s,stream));
    for (int i = 0; i < TOTAL_NUM; i++)
    {
        if(custom_ar == 1)
            custom_all_reduce_comms[rank]->customAllReduce(AR_NUM, stream);
        else
            ftNcclAllReduceSum(dev_buff, dev_buff, AR_NUM, tensor_para, stream);
    }
    check_cuda_error(hipEventRecord(event_e,stream));
    check_cuda_error(hipEventSynchronize(event_e));
    if(custom_ar == 1)
        dev_buff = (half*)(static_cast<CustomAllReduceComm<uint16_t>*>(custom_all_reduce_comms[rank].get())->param_.local_output_buffer_ptr);
    
    // e2e time including cpu time 
    float time_ms;
    check_cuda_error(hipEventElapsedTime(&time_ms,event_s,event_e));
    printf("[rank %d] ElapsedTime : %f ms , For the real time of communication, please use the profile tool\n", rank, time_ms / TOTAL_NUM);
     
    // check the result
    bool flag = true;
    check_cuda_error(hipMemcpyDtoH(&host_buff, dev_buff, sizeof(uint16_t)*AR_NUM));    
    for(int i = 0; i< AR_NUM; i++) 
        if (world_size*1.0 != __half2float(host_buff[i]))
            flag = false;
    if(flag == true)
        printf("[rank %d] check the result : success\n", rank);
    else
        printf("[rank %d] check the result : fail\n", rank);
   
    check_cuda_error(hipDeviceSynchronize());
    
    return 1;
}

#define INSTANTIATE_GEMM_AR_TEST(TA, TB, TScale, TC, TCompute) \
    template int gemm_ar<TA, TB, TScale, TC, TCompute>(const test_args_t& args, const int& rank, const int& world_size);

INSTANTIATE_GEMM_AR_TEST(BHalf, int8_t, float, BHalf, BHalf);

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank , world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    printf("rank=%d, world_size=%d\n", rank, world_size);

    if(argc != 6)
    {
        printf("[ERROR] the pass args could be: [m, n, k, tp, dt], "
               "means [m n k] of gemm, tp means card num, dt means datatype, bf16=0,fp16=1\n"
               "e.g. test_gemm_ar 1 8192 2048 4 0 \n");
        MPI_Finalize();
        return 0;
    }

    test_args_t test_args{static_cast<size_t>(atoi(argv[1])), static_cast<size_t>(atoi(argv[2])), static_cast<size_t>(atoi(argv[3])), static_cast<size_t>(atoi(argv[4])), static_cast<size_t>(atoi(argv[5]))};
    int res = gemm_ar<BHalf, int8_t, float, BHalf, BHalf>(test_args, rank, world_size);
 
    MPI_Finalize();
    return 0;
}


