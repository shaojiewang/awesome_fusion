#pragma once

#include <memory>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "custom_ar_kernels.h"
#include "hip_utils.h"
#include <vector>

namespace AwesomeFusion {

class AbstractCustomComm {
public:
    AbstractCustomComm()                                                             = default;
    virtual ~AbstractCustomComm()                                                    = default;
    virtual void customAllReduce(size_t elts, hipStream_t stream)                   = 0;
    virtual void enableP2P(int ngpus)                                                = 0;
    //virtual bool swapInternalBuffer(std::vector<Tensor>* tensor_buffer, size_t elts) = 0;
    virtual void
    allocateAndExchangePeerAccessPointer(std::vector<std::shared_ptr<AbstractCustomComm>>* custom_all_reduce_comms) = 0;
};

template<typename T>
class CustomAllReduceComm: public AbstractCustomComm {
public:
    CustomAllReduceComm(size_t rank_size, size_t rank);
    ~CustomAllReduceComm();

    void customAllReduce(size_t elts, hipStream_t stream) override;
    //bool swapInternalBuffer(std::vector<Tensor>* tensor_buffer, size_t elts) override;
    void allocateAndExchangePeerAccessPointer(
        std::vector<std::shared_ptr<AbstractCustomComm>>* custom_all_reduce_comms) override;

    void enableP2P(int ngpus) override;

//private:
    AllReduceParams<T>   param_;
    T*                   tmp_tensor_data_;
    size_t               rank_size_;
    size_t               rank_;
};

template<typename T>
void initCustomAllReduceComm(std::vector<std::shared_ptr<AbstractCustomComm>>* custom_all_reduce_comms,
                             int                                               enable_custom_all_reduce,
                             size_t                                            rank_size);

template<typename T>
struct CustomARCommTypeConverter {
    using Type = uint32_t;
};

template<>
struct CustomARCommTypeConverter<half> {
    using Type = uint16_t;
};

}  // namespace AwesomeFusion
