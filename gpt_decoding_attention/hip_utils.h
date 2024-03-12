#pragma once

#include <hip/hip_runtime.h>

using cudaStream_t = hipStream_t;

static const char* _cudaGetErrorEnum(hipError_t error)
{
    return hipGetErrorString(error);
}

template<typename T>
void check(T result, char const* const func, const char* const file, int const line)
{
    if (result) {
        throw std::runtime_error(std::string("[FT][ERROR] CUDA runtime error: ") + (_cudaGetErrorEnum(result)) + " "
                                 + file + ":" + std::to_string(line) + " \n");
    }
}

#define check_cuda_error(val) check((val), #val, __FILE__, __LINE__)

template<typename... Args>
inline std::string fmtstr(const std::string& format, Args... args)
{
    // This function came from a code snippet in stackoverflow under cc-by-1.0
    //   https://stackoverflow.com/questions/2342162/stdstring-formatting-like-sprintf

    // Disable format-security warning in this function.
#if defined(_MSC_VER)  // for visual studio
#pragma warning(push)
#pragma warning(warning(disable : 4996))
#elif defined(__GNUC__) || defined(__clang__)  // for gcc or clang
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-security"
#endif
    int size_s = std::snprintf(nullptr, 0, format.c_str(), args...) + 1;  // Extra space for '\0'
    if (size_s <= 0) {
        throw std::runtime_error("Error during formatting.");
    }
    auto size = static_cast<size_t>(size_s);
    auto buf  = std::make_unique<char[]>(size);
    std::snprintf(buf.get(), size, format.c_str(), args...);
#if defined(_MSC_VER)
#pragma warning(pop)
#elif defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
    return std::string(buf.get(), buf.get() + size - 1);  // We don't want the '\0' inside
}

inline void syncAndCheck(const char* const file, int const line)
{
    // When FT_DEBUG_LEVEL=DEBUG, must check error
    static char* level_name = std::getenv("FT_DEBUG_LEVEL");
    if (level_name != nullptr) {
        static std::string level = std::string(level_name);
        if (level == "DEBUG") {
            hipDeviceSynchronize();
            hipError_t result = hipGetLastError();
            if (result) {
                throw std::runtime_error(std::string("[FT][ERROR] CUDA runtime error: ") + (_cudaGetErrorEnum(result))
                                         + " " + file + ":" + std::to_string(line) + " \n");
            }
            // FT_LOG_DEBUG(fmtstr("run syncAndCheck at %s:%d", file, line));
        }
    }

#ifndef NDEBUG
    hipDeviceSynchronize();
    hipError_t result = hipGetLastError();
    if (result) {
        throw std::runtime_error(std::string("[FT][ERROR] CUDA runtime error: ") + (_cudaGetErrorEnum(result)) + " "
                                 + file + ":" + std::to_string(line) + " \n");
    }
#endif
}

#define sync_check_cuda_error() syncAndCheck(__FILE__, __LINE__)

