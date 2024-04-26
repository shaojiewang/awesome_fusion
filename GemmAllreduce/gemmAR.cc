#include <hip/hip_runtime_api.h>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/time.h>
#include <vector>
#include "nccl.h"
#include <mpi.h>
#include "custom_ar_comm.h"

// whether to use custom kernel[1] or rccl[0]
const int custom_ar = 1;
// num of elements to do all reduce
const int AR_NUM = 8192;

using namespace fastertransformer;

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank , world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
 
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
    
    // output buff
    half *dev_buff, host_buff[AR_NUM], *tmp;
    hipMalloc((void**)&tmp, AR_NUM*sizeof(uint16_t));
    
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
    hipDeviceSynchronize(); 
     
    // perform all reduce
    hipStream_t stream;
    hipStreamCreate(&stream);
    hipEvent_t event_s, event_e;
    hipEventCreate(&event_s);
    hipEventCreate(&event_e);
    MPI_Barrier(MPI_COMM_WORLD);
    check_cuda_error(hipEventRecord(event_s,stream));
    if(custom_ar == 1)
        custom_all_reduce_comms[rank]->customAllReduce(AR_NUM, stream);
    else
        ftNcclAllReduceSum(dev_buff, dev_buff, AR_NUM, tensor_para, stream);
    check_cuda_error(hipEventRecord(event_e,stream));
    check_cuda_error(hipEventSynchronize(event_e));
    if(custom_ar == 1)
        dev_buff = (half*)(static_cast<CustomAllReduceComm<uint16_t>*>(custom_all_reduce_comms[rank].get())->param_.local_output_buffer_ptr);
    
    // e2e time including cpu time 
    float time_ms;
    check_cuda_error(hipEventElapsedTime(&time_ms,event_s,event_e));
    printf("[rank %d] ElapsedTime : %f ms , For the real time of communication, please use the profile tool\n", rank, time_ms);
     
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
    
    MPI_Finalize();
    return 0;
}


