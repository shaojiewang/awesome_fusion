#pragma once

#include "hip_utils.h"
#include <hip/hip_runtime.h>
#include <mpi.h>
#include <rccl/rccl.h>
#include <stdio.h>
#include <string>

#ifdef BUILD_MULTI_GPU
#define MPICHECK(cmd)                                                                                                  \
    do {                                                                                                               \
        int e = cmd;                                                                                                   \
        if (e != MPI_SUCCESS) {                                                                                        \
            printf("Failed: MPI error %s:%d '%d'\n", __FILE__, __LINE__, e);                                           \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)
#else
#define MPICHECK(cmd) printf("[WARNING] No MPI\n");
#endif

namespace fastertransformer {
#ifdef BUILD_MULTI_GPU
#define NCCLCHECK(cmd)                                                                                                 \
    do {                                                                                                               \
        ncclResult_t r = cmd;                                                                                          \
        if (r != ncclSuccess) {                                                                                        \
            printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(r));                      \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)
#else
#define NCCLCHECK(cmd) printf("[WARNING} No NCCL");
#endif

struct NcclUid {
#ifndef BUILD_MULTI_GPU
    NcclUid(){};
    NcclUid(NcclUid const& uid){};
#else
    ncclUniqueId nccl_uid_;
    NcclUid(){};
    NcclUid(NcclUid const& uid): nccl_uid_(uid.nccl_uid_){};
#endif
};

struct NcclParam {
    int rank_{0};
    int world_size_{1};
#ifdef BUILD_MULTI_GPU
    ncclUniqueId nccl_uid_;
    ncclComm_t   nccl_comm_ = nullptr;
#endif

#ifdef BUILD_MULTI_GPU
    NcclParam(): rank_(0), world_size_(1), nccl_comm_(nullptr){};
    NcclParam(int rank, int world_size): rank_(rank), world_size_(world_size){};
    NcclParam(NcclParam const& param):
        rank_(param.rank_), world_size_(param.world_size_), nccl_uid_(param.nccl_uid_), nccl_comm_(param.nccl_comm_){};
    std::string toString()
    {
        return fmtstr("NcclParam[rank=%d, world_size=%d, nccl_comm=%p]", rank_, world_size_, nccl_comm_);
    }
#else
    NcclParam(): rank_(0), world_size_(1){};
    NcclParam(int rank, int world_size): rank_(rank), world_size_(world_size){};
    NcclParam(NcclParam const& param): rank_(param.rank_), world_size_(param.world_size_){};
    std::string toString()
    {
        return fmtstr("NcclParam[rank=%d, world_size=%d]", rank_, world_size_);
    }
#endif
};

// New APIs
template<typename T>
void ftNcclAllReduceSum(const T* send_buf, T* recv_buf, const int data_size, NcclParam nccl_param, hipStream_t stream);

template<typename T>
void ftNcclAllGather(
    const T* send_buf, T* recv_buf, const int data_size, const int rank, NcclParam nccl_param, hipStream_t stream);

template<typename T>
void ftNcclBroadCast(T* buff, const int data_size, const int root, NcclParam nccl_param, hipStream_t stream);

template<typename T>
void ftNcclRecv(T* recv_buf, const int data_size, const int peer, NcclParam nccl_param, hipStream_t stream);

template<typename T>
void ftNcclSend(const T* send_buf, const int data_size, const int peer, NcclParam nccl_param, hipStream_t stream);

// nccl stream synchronize, abort nccl comms and throw errors when nccl async errors detected
void ftNcclStreamSynchronize(NcclParam tensor_para, NcclParam pipeline_para_, hipStream_t stream);

void ftNcclGroupStart();
void ftNcclGroupEnd();
void ftNcclGetUniqueId(NcclUid& uid);
void ftNcclCommInitRank(NcclParam& param, const int rank, const int world_size, const NcclUid uid);
void ftNcclParamDestroy(NcclParam& param);

void ftNcclInitialize(NcclParam& tensor_para,
                      NcclParam& pipeline_para,
                      const int  tensor_para_size,
                      const int  pipeline_para_size);

size_t getLocalBatchSize(const size_t batch_size, const size_t seq_len, const size_t pipeline_para_size);

}  // namespace fastertransformer

