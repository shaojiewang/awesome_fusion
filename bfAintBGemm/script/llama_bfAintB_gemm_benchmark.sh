#/bin/bash

for m in 1 2048 32768; do
   
if [ 0 = 1 ];
then 
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
fi

    echo "llama 175B tp 8"
    ./build/bfAintBGemm.exe ./build 0 2 $m 4608 12288
    ./build/bfAintBGemm.exe ./build 0 2 $m 12288 1536
    ./build/bfAintBGemm.exe ./build 0 2 $m 8192 12288
    ./build/bfAintBGemm.exe ./build 0 2 $m 12288 4096

done

