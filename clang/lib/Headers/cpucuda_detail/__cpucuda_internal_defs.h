
#ifndef __CPUCUDA_INTERNAL_DEFS_H__
#define __CPUCUDA_INTERNAL_DEFS_H__

#ifdef __cplusplus
extern "C" {
#endif

  dim3 __cpucuda_coerced_args_to_dim3(dim3 d) {
    return d;
  }

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
		  size_t shared_mem) {};

  cudaError_t __cpucudaLaunchKernel(
		  //const void* func,
		  int func,
      dim3 grid_dim,
      dim3 block_dim,
      void** args,
      size_t shared_mem,
		  cudaStream_t stream) {};

#ifdef __cplusplus
}
#endif

#endif
