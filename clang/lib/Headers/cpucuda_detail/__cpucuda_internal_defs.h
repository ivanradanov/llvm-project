
#ifndef CPUCUDA_INTERNAL_DEFS_H
#define CPUCUDA_INTERNAL_DEFS_H

/*
#include "types.hpp"

#define __constant__ __attribute__((constant))
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))
#define __device__ __attribute__((device))
#define __host__ __attribute__((host))

#include "../cuda_runtime.h"

#include <stdio.h>

using cpucuda::dim3;
*/

extern "C" {

  unsigned __cpucudaPushCallConfiguration(dim3 gridDim,
                                          dim3 blockDim,
                                          size_t sharedMem = 0,
                                          cudaStream_t stream = 0);

  void __cpucuda_call_kernel(
      //const void* func,
		  int func,
      dim3 grid_dim,
      dim3 block_idx,
      dim3 block_dim,
      void** args,
      size_t shared_mem);

  __host__ cudaError_t __cpucudaLaunchKernel(
		  //const void* func,
		  int func,
      dim3 grid_dim,
      dim3 block_dim,
      void** args,
      size_t shared_mem,
      cudaStream_t stream)
  {
	  /*
    printf("cudaLaunchKernel %i,  %u, %u, %u,  %u, %u, %u,  %p,  %zi,  %i\n",
           func, grid_dim.x, grid_dim.y, grid_dim.z, block_dim.x, block_dim.y, block_dim.z, args, shared_mem, stream);
	  */
    auto execution_stream = _cpucuda_runtime._streams.get(stream);
    (*execution_stream)([=](){
        std::lock_guard<std::mutex> lock{_cpucuda_runtime.dev()._kernel_execution_mutex};

        /*
          printf("in stream %i,  %u, %u, %u,  %u, %u, %u,  %p,  %zi,  %i\n",
          func, grid_dim.x, grid_dim.y, grid_dim.z, block_dim.x, block_dim.y, block_dim.z, args, shared_mem, stream);

          printf("args: %p %p %p %i\n", *((float **)args[0]),*((float **)args[1]),*((float **)args[2]),*((int *)args[3]));
        */

        #pragma omp parallel for collapse(3)
        for(unsigned g_z = 0; g_z < grid_dim.z; ++g_z){
          //printf("g_z %u < %u\n", g_z, grid_dim.z);
          for(unsigned g_y = 0; g_y < grid_dim.y; ++g_y){
            //printf("g_y %u < %u\n", g_y, grid_dim.y);
            for(unsigned g_x = 0; g_x < grid_dim.x; ++g_x){
              //printf("g_x %u < %u\n", g_x, grid_dim.x);
              dim3 block_idx = dim3{g_x, g_y, g_z};
              //printf("in stream %i,  %u, %u, %u\n",
              //func, block_idx.x, block_idx.y, block_idx.z);
              __cpucuda_call_kernel(func, grid_dim, block_idx, block_dim, args, shared_mem);
            }
          }
        }
      });
    return cudaSuccess;
  }

}

#endif
