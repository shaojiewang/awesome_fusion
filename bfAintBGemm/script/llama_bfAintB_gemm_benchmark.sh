#/bin/bash

cd build/

for m in 1 1024 2048; do
    
    echo "llama 13B tp 1"
    ./bfAintBGemm.exe 0 $m 5120 5120
    ./bfAintBGemm.exe 0 $m 15360 5120
    ./bfAintBGemm.exe 0 $m 27648 5120
    ./bfAintBGemm.exe 0 $m 5120 13824

    echo "llama 65B tp 4"
    ./bfAintBGemm.exe 0 $m 6144 8192
    ./bfAintBGemm.exe 0 $m 8192 2048
    ./bfAintBGemm.exe 0 $m 11008 8192
    ./bfAintBGemm.exe 0 $m 8192 5504

done

cd ../

