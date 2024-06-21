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
#include "bfA_intB_gemm_runner.hpp"
#include "validation.hpp"

#define PRINT_BUFFER 1

// whether to use custom kernel[1] or rccl[0]
const int custom_ar = 1;
// num of elements to do all reduce
const int AR_NUM = 256 * 1024;

#define TOTAL_NUM 100
#define WARM_UP_NUM 10

#define MAX_WORLD_SIZE 8
#define MAX_HANDLE_NUM 8

#define BARRIER_FLAG 326
#define MAX_AR_BLOCKS 8

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
    uint32_t m = args.m;
    uint32_t n = args.n;
    uint32_t k = args.k;
    uint32_t tp = args.tp;
    uint32_t k_per_card = k / tp;

    uint32_t lda = k_per_card;
    uint32_t ldb = n;
    uint32_t ldc = n;

    uint32_t lda_ref = k;
    uint32_t ldb_ref = n;
    uint32_t ldc_ref = n;
    
    // assertion
    if (k % (tp * 64) != 0) return 0;
    if (world_size > MAX_WORLD_SIZE) return 0; 

    // initialize custom all reduce 
    std::vector<std::shared_ptr<AbstractCustomComm>> custom_all_reduce_comms;
    initCustomAllReduceComm<hip_bfloat16>(&custom_all_reduce_comms, custom_ar, world_size);
    
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

    // local flags
    DeviceMemCached local_compute_flags(sizeof(int) * ((n + 511) / 512 * 512));

    // multi gpu flags
    DeviceMemUncached multigpu_barrier_flags(sizeof(int) * MAX_WORLD_SIZE * (MAX_AR_BLOCKS + 1));
    
    DeviceMemCached a_device_buf_compute(sizeof(ADataType) * f_matrix_space_size(m, k_per_card, lda, ALayout{}));
    DeviceMemCached b_device_buf_compute(sizeof(BDataType) * f_matrix_space_size(k_per_card, n, ldb, BLayout{}));

    DeviceMemCached a_device_buf(sizeof(ADataType) * f_matrix_space_size(m, k_per_card, lda, ALayout{}));
    DeviceMemCached b_device_buf(sizeof(BDataType) * f_matrix_space_size(k_per_card, n, ldb, BLayout{}));
    DeviceMemUncached c_device_buf(sizeof(CDataType) * f_matrix_space_size(m, n, ldc, CLayout{}));
    DeviceMemCached c_device_buf_out(sizeof(CDataType) * f_matrix_space_size(m, n, ldc, CLayout{}));
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


    // pointer for barrier flags
    void* multigpu_barrier_flag_ptrs[MAX_WORLD_SIZE];

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
            multigpu_barrier_flag_ptrs[i] = reinterpret_cast<void*>(multigpu_barrier_flags.GetBuffer());
            check_cuda_error(hipIpcGetMemHandle(&(handle[6]), multigpu_barrier_flag_ptrs[i]));
        }
        MPI_Bcast(&(handle[0]), sizeof(hipIpcMemHandle_t), MPI_CHAR, i, MPI_COMM_WORLD);
        MPI_Bcast(&(handle[1]), sizeof(hipIpcMemHandle_t), MPI_CHAR, i, MPI_COMM_WORLD);
        MPI_Bcast(&(handle[2]), sizeof(hipIpcMemHandle_t), MPI_CHAR, i, MPI_COMM_WORLD);
        MPI_Bcast(&(handle[3]), sizeof(hipIpcMemHandle_t), MPI_CHAR, i, MPI_COMM_WORLD);
        MPI_Bcast(&(handle[4]), sizeof(hipIpcMemHandle_t), MPI_CHAR, i, MPI_COMM_WORLD);
        MPI_Bcast(&(handle[5]), sizeof(hipIpcMemHandle_t), MPI_CHAR, i, MPI_COMM_WORLD);
        MPI_Bcast(&(handle[6]), sizeof(hipIpcMemHandle_t), MPI_CHAR, i, MPI_COMM_WORLD);
        if (rank != i)
        {
            check_cuda_error(hipIpcOpenMemHandle((void **)&(init_a_buf_ptrs[i]), handle[0], hipIpcMemLazyEnablePeerAccess));
            check_cuda_error(hipIpcOpenMemHandle((void **)&(init_a_buf_ref_ptrs[i]), handle[1], hipIpcMemLazyEnablePeerAccess));
            check_cuda_error(hipIpcOpenMemHandle((void **)&(init_b_buf_ptrs[i]), handle[2], hipIpcMemLazyEnablePeerAccess));
            check_cuda_error(hipIpcOpenMemHandle((void **)&(init_b_buf_ref_ptrs[i]), handle[3], hipIpcMemLazyEnablePeerAccess));
            check_cuda_error(hipIpcOpenMemHandle((void **)&(init_scale_buf_ptrs[i]), handle[4], hipIpcMemLazyEnablePeerAccess));
            check_cuda_error(hipIpcOpenMemHandle((void **)&(init_scale_buf_ref_ptrs[i]), handle[5], hipIpcMemLazyEnablePeerAccess));
            check_cuda_error(hipIpcOpenMemHandle((void **)&(multigpu_barrier_flag_ptrs[i]), handle[6], hipIpcMemLazyEnablePeerAccess));
        }
        
    }

    // init tensor on rank 0
    if (rank == 0)
    {
        cudaRandomUniform<ADataType>(reinterpret_cast<ADataType*>(a_device_buf_ref.GetBuffer()), m * k);
        cudaRandomUniform<BDataType>(reinterpret_cast<BDataType*>(b_device_buf_ref.GetBuffer()), n * k);
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
    MPI_Barrier(MPI_COMM_WORLD);

    // fix compute b ref in bfloat16
    invokeMatrixElementwiseScale(reinterpret_cast<hip_bfloat16*>(b_device_buf_ref_compute.GetBuffer()), reinterpret_cast<int8_t*>(b_device_buf_ref.GetBuffer()), reinterpret_cast<float*>(scale_device_buf_ref.GetBuffer()), n, k);

    // reference result by rocblas
    ADataType* d_A = reinterpret_cast<ADataType*>(init_a_buf_ref_ptrs[rank]);
    // ADataType* d_A = reinterpret_cast<ADataType*>(a_device_buf_ref.GetBuffer());
    ComputeDataType* d_B = reinterpret_cast<ComputeDataType*>(b_device_buf_ref_compute.GetBuffer());
    // printf("in rank %d, b buf compute = 0x%x\n", rank, *(int*)b_device_buf_ref_compute.GetBuffer());
    CDataType* d_C = reinterpret_cast<CDataType*>(c_device_buf_ref.GetBuffer());
    TGemm<ADataType> gemm_t(m, n, k, d_A, d_B, d_C, true, false);
    call_rocBLAS(gemm_t, reinterpret_cast<CDataType*>(c_host_buf.GetBuffer()));

    check_cuda_error(hipDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    printf("rank %d, c_ref is [%f]\n", rank, type_convert<float, hip_bfloat16>(reinterpret_cast<hip_bfloat16*>(c_device_buf_ref.GetBuffer())[2]));

    MPI_Barrier(MPI_COMM_WORLD);
    // check broadcast res
#if PRINT_BUFFER
    if (rank == 0)
    {
        printf("a buf ref is [0x%x, 0x%x, 0x%x, 0x%x]\n", 
            *(int*)(init_a_buf_ref_ptrs[0]),
            *(int*)((char*)(init_a_buf_ref_ptrs[0]) + sizeof(ADataType) * m * k_per_card),
            *(int*)((char*)(init_a_buf_ref_ptrs[0]) + 2 * sizeof(ADataType) * m * k_per_card),
            *(int*)((char*)(init_a_buf_ref_ptrs[0]) + 3 * sizeof(ADataType) * m * k_per_card));
        printf("b buf ref is [0x%x, 0x%x, 0x%x, 0x%x]\n", 
            *(int*)(init_b_buf_ref_ptrs[0]),
            *(int*)((char*)(init_b_buf_ref_ptrs[0]) + sizeof(BDataType) * n * k_per_card),
            *(int*)((char*)(init_b_buf_ref_ptrs[0]) + 2 * sizeof(BDataType) * n * k_per_card),
            *(int*)((char*)(init_b_buf_ref_ptrs[0]) + 3 * sizeof(BDataType) * n * k_per_card));
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
#endif

    hipStream_t communication_stream;
    check_cuda_error(hipStreamCreate(&communication_stream));

    hipStream_t compute_stream;
    check_cuda_error(hipStreamCreate(&compute_stream));

    // add matrix transpose code
    // 1. transpose A matrix
    invokeMatrixTranspose(reinterpret_cast<hip_bfloat16*>(a_device_buf_compute.GetBuffer()), reinterpret_cast<hip_bfloat16*>(a_device_buf.GetBuffer()), m, k_per_card, compute_stream);
    // 2. transpose and interleave B matrix
    invokeMatrixBatchedTranspose<uint8_t>(reinterpret_cast<uint8_t*>(b_device_buf_compute.GetBuffer()), reinterpret_cast<uint8_t*>(b_device_buf.GetBuffer()), 16, n, k_per_card / 16, compute_stream);
    
    check_cuda_error(hipDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

#if PRINT_BUFFER
    // check A transpose
    printf("a_device_buf_compute=[%x, %x, %x, %x]\n", 
        reinterpret_cast<int*>(a_device_buf_compute.GetBuffer())[12],
        reinterpret_cast<int*>(a_device_buf_compute.GetBuffer())[13],
        reinterpret_cast<int*>(a_device_buf_compute.GetBuffer())[14],
        reinterpret_cast<int*>(a_device_buf_compute.GetBuffer())[15]);
    printf("a_device_buf=[%x]\n", reinterpret_cast<int*>(a_device_buf.GetBuffer())[0]);
    // check B transpose
    printf("b_device_buf_compute=[%x, %x, %x, %x, %x, %x, %x, %x, %x, %x, %x, %x, %x, %x, %x, %x]\n", 
        reinterpret_cast<int*>(b_device_buf_compute.GetBuffer())[0],
        reinterpret_cast<int*>(b_device_buf_compute.GetBuffer())[1],
        reinterpret_cast<int*>(b_device_buf_compute.GetBuffer())[2],
        reinterpret_cast<int*>(b_device_buf_compute.GetBuffer())[3],
        reinterpret_cast<int*>(b_device_buf_compute.GetBuffer())[n * 4],
        reinterpret_cast<int*>(b_device_buf_compute.GetBuffer())[n * 4 + 1],
        reinterpret_cast<int*>(b_device_buf_compute.GetBuffer())[n * 4 + 2],
        reinterpret_cast<int*>(b_device_buf_compute.GetBuffer())[n * 4 + 3],
        reinterpret_cast<int*>(b_device_buf_compute.GetBuffer())[n * 8],
        reinterpret_cast<int*>(b_device_buf_compute.GetBuffer())[n * 8 + 1],
        reinterpret_cast<int*>(b_device_buf_compute.GetBuffer())[n * 8 + 2],
        reinterpret_cast<int*>(b_device_buf_compute.GetBuffer())[n * 8 + 3],
        reinterpret_cast<int*>(b_device_buf_compute.GetBuffer())[n * 12],
        reinterpret_cast<int*>(b_device_buf_compute.GetBuffer())[n * 12 + 1],
        reinterpret_cast<int*>(b_device_buf_compute.GetBuffer())[n * 12 + 2],
        reinterpret_cast<int*>(b_device_buf_compute.GetBuffer())[n * 12 + 3]);

    printf("b_device_buf=[%x, %x, %x, %x, %x, %x, %x]\n",
        reinterpret_cast<int*>(b_device_buf.GetBuffer())[0],
        reinterpret_cast<int*>(b_device_buf.GetBuffer())[n / 4],
        reinterpret_cast<int*>(b_device_buf.GetBuffer())[n / 2],
        reinterpret_cast<int*>(b_device_buf.GetBuffer())[n / 4 * 3],
        reinterpret_cast<int*>(b_device_buf.GetBuffer())[n * 4],
        reinterpret_cast<int*>(b_device_buf.GetBuffer())[n * 8],
        reinterpret_cast<int*>(b_device_buf.GetBuffer())[n * 12]);

    // check scale 
    printf("scale buffer=[%f, %f, %f, %f]\n", 
        reinterpret_cast<float*>(scale_device_buf.GetBuffer())[0], 
        reinterpret_cast<float*>(scale_device_buf.GetBuffer())[1],
        reinterpret_cast<float*>(scale_device_buf.GetBuffer())[2],
        reinterpret_cast<float*>(scale_device_buf.GetBuffer())[3]);
#endif

#ifdef ASM_PRINT
    //debug pointer
    float *host_print, *print;
    uint32_t print_sk_blocks = 1;
    host_print = (float*)malloc(1024*8);
    check_cuda_error(hipMalloc(&print, 1024*8));
#endif

    // gemm + ar reference
    // get kernel list
    std::vector<kernel_tunable> k_list = get_kernel_list();
    std::string hsaco_path = "./build/";
    uint32_t max_sk_blocks = 1;
    uint32_t barrier_flag = BARRIER_FLAG;
    bfAintBGemmRunner bfa_intb_gemm_runner(k_list,
                                           hsaco_path,  
                                           c_device_buf.GetBuffer(),
                                           a_device_buf_compute.GetBuffer(),
                                           b_device_buf_compute.GetBuffer(),
                                           scale_device_buf.GetBuffer(),
                                           m,
                                           n,
                                           k_per_card,
                                           lda,
                                           ldb,
                                           ldc,
                                           k_per_card,
#ifdef ASM_PRINT
                                           print,
                                           print_sk_blocks
#else
                                           nullptr,
                                           max_sk_blocks
#endif
      ,
                                           barrier_flag,
                                           local_compute_flags.GetBuffer(),
                                           multigpu_barrier_flag_ptrs,
                                           nullptr,
                                           nullptr,
                                           (size_t)rank
                                           );
    printf("multigpu_barrier_flag_ptrs=%p\n", multigpu_barrier_flag_ptrs);

    // ar init
    if(custom_ar == 1)
    {
        static_cast<CustomAllReduceComm<hip_bfloat16>*>(custom_all_reduce_comms[rank].get())->param_.local_output_buffer_ptr = reinterpret_cast<hip_bfloat16*>(c_device_buf_out.GetBuffer());
        static_cast<CustomAllReduceComm<hip_bfloat16>*>(custom_all_reduce_comms[rank].get())->param_.peer_comm_buffer_ptrs[rank] = reinterpret_cast<hip_bfloat16*>(c_device_buf.GetBuffer());
        // re-broadcast
        for (int i = 0; i < world_size; i++)
        {
            hipIpcMemHandle_t handle;
            if (rank == i)
            {
                check_cuda_error(hipIpcGetMemHandle(&handle,
                                                    static_cast<CustomAllReduceComm<hip_bfloat16>*>(custom_all_reduce_comms[rank].get())->param_.peer_comm_buffer_ptrs[rank]));
            }
            MPI_Bcast(&handle, sizeof(hipIpcMemHandle_t), MPI_CHAR, i, MPI_COMM_WORLD);
            if (rank != i)
            {
                check_cuda_error(hipIpcOpenMemHandle((void **)&(static_cast<CustomAllReduceComm<hip_bfloat16>*>(custom_all_reduce_comms[rank].get())->param_.peer_comm_buffer_ptrs[i]), handle, hipIpcMemLazyEnablePeerAccess));
            }
        }
        
    }
    else
    {
        // dev_buff = tmp;
    }

    check_cuda_error(hipDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    uint32_t sol_idx = 0, sk_blocks = 1;
    
    for(int i = 0; i < WARM_UP_NUM; i++)
    {
        bfa_intb_gemm_runner.run(bfa_intb_gemm_runner.k_ptr[sol_idx], bfa_intb_gemm_runner.kernel_func_vec[sol_idx], nullptr, sk_blocks);
        custom_all_reduce_comms[rank]->customAllReduce(m * n, nullptr);
    }

    hipEvent_t evt_00, evt_11;
    float elapsed_ms;
    check_cuda_error(hipEventCreate(&evt_00));
    check_cuda_error(hipEventCreate(&evt_11));
    check_cuda_error(hipDeviceSynchronize());
    check_cuda_error(hipEventRecord(evt_00, compute_stream));

    for(int i = 0; i < TOTAL_NUM; i++)
    {
        bfa_intb_gemm_runner.run(bfa_intb_gemm_runner.k_ptr[sol_idx], bfa_intb_gemm_runner.kernel_func_vec[sol_idx], compute_stream, sk_blocks);
        custom_all_reduce_comms[rank]->customAllReduce(m * n, compute_stream);
    }

    check_cuda_error(hipEventRecord(evt_11, compute_stream));
    check_cuda_error(hipEventSynchronize(evt_11));
    check_cuda_error(hipDeviceSynchronize());
    check_cuda_error(hipEventElapsedTime(&elapsed_ms, evt_00, evt_11));
    check_cuda_error(hipEventDestroy(evt_00));
    check_cuda_error(hipEventDestroy(evt_11));

    // check bf16 gemm res
#if PRINT_BUFFER
    printf("rank %d, res=%f\n", 
        rank, 
        type_convert<float, hip_bfloat16>(reinterpret_cast<hip_bfloat16*>(c_device_buf.GetBuffer())[29]));
#endif

#ifdef ASM_PRINT
    if (rank == 1)
    {
        int max_i = bfa_intb_gemm_runner.k_ptr[sol_idx].wg_size;
        check_cuda_error(hipMemcpy(host_print, print, 8*max_i, hipMemcpyDeviceToHost));
        for(int i = 0; i < max_i; i++){
            // if(((uint32_t*)host_print)[2*i+1]!=0x5c005c00)
            float fp32_val = ((float*)host_print)[2*i+1];
            uint32_t fp32_val_bit = __builtin_bit_cast(uint32_t, fp32_val);
            float bf16_lo = __builtin_bit_cast(float, (fp32_val_bit << 16));
            float bf16_hi = __builtin_bit_cast(float, (fp32_val_bit & 0xffff0000));
            printf("%dth: Thread%d, PrintVal:0x%x, %d, %f, [%f, %f]\n", i, ((int*) host_print)[2*i], fp32_val_bit, fp32_val_bit, fp32_val, bf16_lo, bf16_hi);
            //std::cout<<"Thread"<<((int*) host_print)[2*i]<<", PrintVal1:"<<(((float16*)host_print)[4*i+2])<<
            //", PrintVal2:"<<( ( (float16*)host_print )[4*i+3] )<<std::endl;
        }
    }    
#endif

    float time_per_loop = elapsed_ms / TOTAL_NUM;
    float tflops = (float)2 * m * n * k_per_card / time_per_loop / (1024 * 1024 * 1024);
    float bw_gbs = (float)(2 * (m * k_per_card + m * n) + n * k_per_card) / time_per_loop / (1024 * 1024);
    
    printf("best [sol, sk_blocks]: [%d, %d], m: %d, n: %d, k: %d, time: %.3f ms, tflops: %.3f, bw: %.3f GB/s\n",
        sol_idx, sk_blocks,
        m,
        n,
        k_per_card,
        time_per_loop,
        tflops,
        bw_gbs);
    printf("\n");

    // result checker
    valid_vector<hip_bfloat16>(reinterpret_cast<hip_bfloat16*>(c_device_buf_ref.GetBuffer()), 
                               reinterpret_cast<hip_bfloat16*>(c_device_buf_out.GetBuffer()),
                               m * n);
    return 0;

    // output buff
    hip_bfloat16 *dev_buff, host_buff[AR_NUM], *tmp;
    check_cuda_error(hipMalloc((void**)&tmp, AR_NUM*sizeof(hip_bfloat16)));
    
    if(custom_ar == 1){
        static_cast<CustomAllReduceComm<hip_bfloat16>*>(custom_all_reduce_comms[rank].get())->param_.local_output_buffer_ptr = (hip_bfloat16*)tmp;
        dev_buff = (hip_bfloat16*)static_cast<CustomAllReduceComm<hip_bfloat16>*>(custom_all_reduce_comms[rank].get())->param_.peer_comm_buffer_ptrs[rank];
    }
    else{
        dev_buff = tmp;
    }

    // replaced with ops like ffn
    for(int i = 0; i< AR_NUM; i++){
        host_buff[i] = type_convert<hip_bfloat16, float>(1.0);
    }
    check_cuda_error(hipMemcpyHtoD(dev_buff, &host_buff, sizeof(hip_bfloat16)*AR_NUM));
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
        dev_buff = (hip_bfloat16*)(static_cast<CustomAllReduceComm<hip_bfloat16>*>(custom_all_reduce_comms[rank].get())->param_.local_output_buffer_ptr);
    
    // e2e time including cpu time 
    float time_ms;
    check_cuda_error(hipEventElapsedTime(&time_ms,event_s,event_e));
    printf("[rank %d] ElapsedTime : %f ms , For the real time of communication, please use the profile tool\n", rank, time_ms / TOTAL_NUM);
     
    // check the result
    bool flag = true;
    check_cuda_error(hipMemcpyDtoH(&host_buff, dev_buff, sizeof(hip_bfloat16)*AR_NUM));    
    for(int i = 0; i< AR_NUM; i++) 
        if (world_size*1.0 != type_convert<float, hip_bfloat16>(host_buff[i]))
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


