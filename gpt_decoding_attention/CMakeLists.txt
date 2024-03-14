cmake_minimum_required(VERSION 3.16)

# enable_language(HIP)
project(ft_decoder_paged_attn VERSION 1.0.0 LANGUAGES CXX)

# Find HIP
list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)
find_package(HIP REQUIRED)

# Set compilation standard
set(CMAKE_CLANG_STANDARD 17)
set(CMAKE_CLANG_STANDARD_REQUIRED ON)
set(CMAKE_CLANG_EXTENSIONS OFF) # https://www.reddit.com/user

# Build HIP depending on platform
message(STATUS "HIP_PLATFORM: amd")
add_compile_definitions(__HIP_PLATFORM_AMD__)
add_definitions(-DENABLE_BF16=1 -DVIEW_ERR_POINT=1 -DINT_INIT=1 -DENABLE_INT8=0)
add_definitions(-DENABLE_MULTI_BLOCK_OPTION )
add_compile_options(--save-temps)
set(CMAKE_HIP_ARCHITECTURES "gfx90a")

# Library:
set(CU_SRCS
    unfused_attention_kernels_ie_rocm.cc
    transpose_kv_cache.cc
    decoder_masked_multihead_attention.cc 
    decoder_paged_masked_multihead_attention.cc)
# set_source_files_properties(decoder_masked_multihead_attention.cu PROPERTIES LANGUAGE HIP)
add_library(hip_decoder_attn_lib SHARED ${CU_SRCS})

# Executables:
add_executable(test_decoder_attention test_decoder_attention.cc)
target_link_libraries(test_decoder_attention PUBLIC hip_decoder_attn_lib)
# Executables:
add_executable(test_paged_decoder_attention test_paged_decoder_attention.cc)
target_link_libraries(test_paged_decoder_attention PUBLIC hip_decoder_attn_lib)
