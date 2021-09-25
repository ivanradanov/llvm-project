
#ifndef CPUCUDA_INTERNAL_DEFS_H
#define CPUCUDA_INTERNAL_DEFS_H

#include "types.hpp"

#define __constant__ __attribute__((constant))
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))
#define __device__ __attribute__((device))
#define __host__ __attribute__((host))

#include "../cuda_runtime.h"

using cpucuda::dim3;

extern "C" {

  dim3 __cpucuda_construct_dim3(unsigned x, unsigned y, unsigned z) {
    return dim3(x, y, z);
  }
  unsigned __cpucuda_dim3_get_x(dim3 d) {
    return d.x;
  }
  unsigned __cpucuda_dim3_get_y(dim3 d) {
    return d.y;
  }
  unsigned __cpucuda_dim3_get_z(dim3 d) {
    return d.z;
  }

  dim3 __cpucuda_coerced_args_to_dim3(dim3 d) {
    return d;
  }

  dim3 __cpucuda_declared_dim3_getter();
  void __cpucuda_declared_dim3_user(dim3 d);

  void __cpucuda_dim3_to_arg() {
    dim3 d = __cpucuda_declared_dim3_getter();
    __cpucuda_declared_dim3_user(d);
  }

  dim3 __cpucuda_threadIdx();
  dim3 __cpucuda_blockIdx();
  dim3 __cpucuda_blockDim();
  dim3 __cpucuda_gridDim();
  void __cpucuda_syncthreads();

  dim3 __cpucuda_real_blockIdx();
  dim3 __cpucuda_real_blockDim();
  dim3 __cpucuda_real_gridDim();

//#define __shared__ __attribute__((annotate("cpucuda_shared")))

//extern "C" __host__ __device__  unsigned CUDARTAPI __cudaPushCallConfiguration(dim3 gridDim,
  unsigned __cpucudaPushCallConfiguration(dim3 gridDim,
                                          dim3 blockDim,
                                          size_t sharedMem = 0,
                                          cudaStream_t stream = 0);

  /*
  void __cpucuda_placeholder_kernel(dim3 griddim, dim3 blockidx, dim3 blockdim, size_t sharedMem = 0);

  void __cpucuda_submit_kernel(dim3 *_grid, dim3 *_block, int shared_mem, int stream)
  {
    dim3 grid = *_grid;
    dim3 block = *_block;
    auto execution_stream = _cpucuda_runtime._streams.get(stream);
    (*execution_stream)([=](){
        std::lock_guard<std::mutex> lock{_cpucuda_runtime.dev()._kernel_execution_mutex};

#pragma omp parallel for collapse(3)
        for(size_t g_x = 0; g_x < grid.x; ++g_x){
          for(size_t g_y = 0; g_y < grid.y; ++g_y){
            for(size_t g_z = 0; g_z < grid.z; ++g_z){
              dim3 block_idx = dim3{g_x, g_y, g_z};
              __cpucuda_placeholder_kernel(grid, block_idx, block, shared_mem);
            }
          }
        }
      });
  }
	*/

  void __cpucuda_call_kernel(
    const void* func,
    dim3 grid_dim,
    dim3 block_dim,
    dim3 block_idx,
    void** args,
    size_t shared_mem);

  __host__ cudaError_t cudaLaunchKernel(const void* func,
                                        dim3 grid_dim,
                                        dim3 block_dim,
                                        void** args,
                                        size_t shared_mem,
                                        cudaStream_t stream)
  {
    auto execution_stream = _cpucuda_runtime._streams.get(stream);
    (*execution_stream)([=](){
        std::lock_guard<std::mutex> lock{_cpucuda_runtime.dev()._kernel_execution_mutex};

#pragma omp parallel for collapse(3)
        for(size_t g_x = 0; g_x < grid_dim.x; ++g_x){
          for(size_t g_y = 0; g_y < grid_dim.y; ++g_y){
            for(size_t g_z = 0; g_z < grid_dim.z; ++g_z){
              dim3 block_idx = dim3{g_x, g_y, g_z};
              __cpucuda_call_kernel(func, grid_dim, block_dim, block_idx, args, shared_mem);
            }
          }
        }
      });


  }

  void __cpucuda_real_func_user() {
    __cpucuda_real_blockIdx();
    __cpucuda_real_blockDim();
    __cpucuda_real_gridDim();
  }

}

#define threadIdx __cpucuda_threadIdx()
#define blockIdx __cpucuda_blockIdx()
#define blockDim __cpucuda_blockDim()
#define gridDim __cpucuda_gridDim()
#define __syncthreads __cpucuda_syncthreads

#endif
