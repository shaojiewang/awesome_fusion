#/bin/bash

for m in 1 1024 2048; do
    
    echo "llama 13B tp 1"
    ./build/bfAintBGemm.exe ./build 0 1 $m 5120 5120
    ./build/bfAintBGemm.exe ./build 0 1 $m 15360 5120
    ./build/bfAintBGemm.exe ./build 0 1 $m 27648 5120
    ./build/bfAintBGemm.exe ./build 0 1 $m 5120 13824

    echo "llama 65B tp 4"
    ./build/bfAintBGemm.exe ./build 0 1 $m 6144 8192
    ./build/bfAintBGemm.exe ./build 0 1 $m 8192 2048
    ./build/bfAintBGemm.exe ./build 0 1 $m 11008 8192
    ./build/bfAintBGemm.exe ./build 0 1 $m 8192 5504

done

