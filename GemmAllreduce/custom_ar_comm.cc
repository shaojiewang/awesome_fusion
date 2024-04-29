#include "custom_ar_comm.h"

namespace AwesomeFusion {

template<typename T>
CustomAllReduceComm<T>::CustomAllReduceComm(size_t rank_size, size_t rank): rank_size_(rank_size), rank_(rank)
{
    param_.barrier_flag = 0;
    param_.rank       = rank_size;
    param_.local_rank = rank_;
    param_.node_id    = 0;
}

template<typename T>
CustomAllReduceComm<T>::~CustomAllReduceComm()
{
    hipPointerAttribute_t comm_buffer_attributes, barrier_attributes;
    check_cuda_error(hipPointerGetAttributes(&comm_buffer_attributes, param_.peer_comm_buffer_ptrs[rank_]));
    check_cuda_error(hipPointerGetAttributes(&barrier_attributes, param_.peer_barrier_ptrs[rank_]));
    if (comm_buffer_attributes.type == 2) {
        check_cuda_error(hipFree(param_.peer_comm_buffer_ptrs[rank_]));
    }
    if (barrier_attributes.type == 2) {
        check_cuda_error(hipFree(param_.peer_barrier_ptrs[rank_]));
    }
}

template<typename T>
void CustomAllReduceComm<T>::customAllReduce(size_t elts, hipStream_t stream)
{
    param_.elts_total   = elts;
    param_.barrier_flag = FLAG(param_.barrier_flag + 1);

    invokeOneOrTwoShotAllReduceKernel<T>(param_, stream);
    // swap back
    // output_tensor_->at(0).data = (const void *)tmp_tensor_data_;;
}

template<typename T>
void CustomAllReduceComm<T>::allocateAndExchangePeerAccessPointer(
    std::vector<std::shared_ptr<AbstractCustomComm>>* custom_all_reduce_comms)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    //std::cout<< rank;
    assert(custom_all_reduce_comms->size() == rank_size_);
    assert(rank_ == 0);
    // Enable Peer to Peer Access
    // enableP2P(rank_size_);
    for (size_t i = 0; i < rank_size_; i++) {
        check_cuda_error(hipSetDevice(i));
        hipIpcMemHandle_t handle;
        if (rank == i){ 
            check_cuda_error(hipExtMallocWithFlags((void **)&(param_.peer_comm_buffer_ptrs[i]), CUSTOM_AR_SIZE_THRESHOLD * sizeof(T), hipDeviceMallocFinegrained));
            //hipMalloc(&(param_.peer_comm_buffer_ptrs[i]), CUSTOM_AR_SIZE_THRESHOLD);
            hipIpcGetMemHandle(&handle,param_.peer_comm_buffer_ptrs[i]);
        }   
        MPI_Bcast(&handle, sizeof(hipIpcMemHandle_t), MPI_CHAR, i, MPI_COMM_WORLD);
        hipIpcOpenMemHandle((void **)&(param_.peer_comm_buffer_ptrs[i]), handle, hipIpcMemLazyEnablePeerAccess);

        hipIpcMemHandle_t handle2;
        if (rank == i){ 
            check_cuda_error(hipExtMallocWithFlags((void **)&(param_.peer_barrier_ptrs[i]),  rank_size_ * (MAX_ALL_REDUCE_BLOCKS + 1) * sizeof(uint32_t), hipDeviceMallocFinegrained));
            //hipMalloc(&(param_.peer_barrier_ptrs[i]), rank_size_ * (MAX_ALL_REDUCE_BLOCKS + 1) * sizeof(uint32_t));
            hipIpcGetMemHandle(&handle2, param_.peer_barrier_ptrs[i]);
        }
        MPI_Bcast(&handle2, sizeof(hipIpcMemHandle_t) , MPI_CHAR, i, MPI_COMM_WORLD);
        hipIpcOpenMemHandle((void **)&(param_.peer_barrier_ptrs[i]), handle2, hipIpcMemLazyEnablePeerAccess);
 
        check_cuda_error(
            hipMemset((void*)param_.peer_barrier_ptrs[i], 0, rank_size_ * (MAX_ALL_REDUCE_BLOCKS + 1) * sizeof(uint32_t)));
        T*        current_peer_comm_buffer_ptr = param_.peer_comm_buffer_ptrs[i];
        uint32_t* current_peer_barrier_ptr     = param_.peer_barrier_ptrs[i];
        // Assume current comm allocates device memory on all ranks (rank_ == 0)
        for (size_t j = 1; j < rank_size_; j++) {
            static_cast<CustomAllReduceComm<T>*>(custom_all_reduce_comms->at(j).get())
                ->param_.peer_comm_buffer_ptrs[i] = current_peer_comm_buffer_ptr;
            static_cast<CustomAllReduceComm<T>*>(custom_all_reduce_comms->at(j).get())->param_.peer_barrier_ptrs[i] =
                current_peer_barrier_ptr;
        }
    }
}

// unecessary for amd
template<typename T>
void CustomAllReduceComm<T>::enableP2P(int ngpus)
{
    int peer_access_available = 0;
    for (int i = 0; i < ngpus; i++) {
        hipSetDevice(i);
        for (int j = 0; j < ngpus; j++) {
            if (i == j) {
                continue;
            }
            hipDeviceCanAccessPeer(&peer_access_available, i, j);\
            assert(peer_access_available);
            hipDeviceEnablePeerAccess(j, 0);
        }
    }
}

template<typename T>
void initCustomAllReduceComm(std::vector<std::shared_ptr<AbstractCustomComm>>* custom_all_reduce_comms,
                             int                                               enable_custom_all_reduce,
                             size_t                                            rank_size)
{
    if (custom_all_reduce_comms == 0 || !enable_custom_all_reduce) {
        for (size_t i = 0; i < rank_size; i++) {
            custom_all_reduce_comms->push_back(nullptr);
        }
        return;
    }

    for (size_t i = 0; i < rank_size; i++) {
        custom_all_reduce_comms->push_back(std::make_shared<CustomAllReduceComm<T>>(rank_size, i));
    }
    custom_all_reduce_comms->at(0)->allocateAndExchangePeerAccessPointer(custom_all_reduce_comms);
}

/*
template<typename T>
bool CustomAllReduceComm<T>::swapInternalBuffer(std::vector<Tensor>* tensor_buffer, size_t elts)
{

    if (rank_size_ > 1 && elts * sizeof(T) <= CUSTOM_AR_SIZE_THRESHOLD) {
        tmp_tensor_data_               = (T*)(tensor_buffer->at(0).data);
        output_tensor_                 = tensor_buffer;
        tensor_buffer->at(0).data      = param_.peer_comm_buffer_ptrs[rank_];
        param_.local_output_buffer_ptr = tmp_tensor_data_;
        return true;
    }
    return false;
}
*/

template class CustomAllReduceComm<uint16_t>;
template class CustomAllReduceComm<uint32_t>;

template void
initCustomAllReduceComm<uint16_t>(std::vector<std::shared_ptr<AbstractCustomComm>>* custom_all_reduce_comms,
                                  int                                               enable_custom_all_reduce,
                                  size_t                                            rank_size);

template void
initCustomAllReduceComm<uint32_t>(std::vector<std::shared_ptr<AbstractCustomComm>>* custom_all_reduce_comms,
                                  int                                               enable_custom_all_reduce,
                                  size_t                                            rank_size);
}  // namespace AwesomeFusion
