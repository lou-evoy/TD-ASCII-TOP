/* CUDA error-checking helpers for the ASCII TOP.
 *
 * Kernels launch asynchronously, so most errors only surface at a later synchronizing
 * call. We check every synchronous CUDA call inline and cudaGetLastError() after each
 * launch. Nothing aborts: the algorithm layer returns a cudaError_t and a message to
 * the TD glue, which puts the node into a clean error state via getErrorString().
 */
#ifndef ASCII_CUDA_CHECK_H
#define ASCII_CUDA_CHECK_H

#include "cuda_runtime.h"
#include <cstdio>

#define AX_CUDA_RETURN(expr, outErrPtr)                                          \
    do {                                                                         \
        cudaError_t ax_err__ = (expr);                                           \
        if (ax_err__ != cudaSuccess) {                                           \
            ax_setError((outErrPtr), #expr, ax_err__, __FILE__, __LINE__);       \
            return ax_err__;                                                     \
        }                                                                        \
    } while (0)

#define AX_CUDA_CHECK_LAUNCH(outErrPtr)  AX_CUDA_RETURN(cudaGetLastError(), (outErrPtr))

inline char* ax_errorBuffer()
{
    static char buf[512];
    return buf;
}

inline void ax_setError(const char** outErrPtr, const char* expr,
                        cudaError_t err, const char* file, int line)
{
    char* buf = ax_errorBuffer();
    snprintf(buf, 512, "CUDA error %d (%s) at %s:%d -> %s",
             (int)err, cudaGetErrorString(err), file, line, expr);
#ifdef _DEBUG
    fprintf(stderr, "[AsciiTOP] %s\n", buf);
#endif
    if (outErrPtr)
        *outErrPtr = buf;
}

#endif // ASCII_CUDA_CHECK_H
