# GEMM + AllReduce fusion kernel development
In llm inference and training, all reduce is a critical communication proto for the end to end elapsed time. We noticed that the all reduce always follow a gemm. So in this repo, we will do some trails to fuse these 2 component.

## algorithm
``` python
counter_idx = bidx // 4
counter_num = atomic_fetch_add(counter_ptr[counter_idx], 1)
if counter_num == bidx // 4 - 1:
    # multi gpu sync
    
    # read all cards data

    # perform reduce

```
