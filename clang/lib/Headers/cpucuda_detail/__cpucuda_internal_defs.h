
#ifndef __CPUCUDA_INTERNAL_DEFS_H__
#define __CPUCUDA_INTERNAL_DEFS_H__

#ifdef __cplusplus
extern "C" {
#endif
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
  dim3 __cpucuda_dim3ptr_to_dim3(dim3 *d) {
    return *d;
  }
  dim3 __cpucuda_declared_dim3_getter();
  void __cpucuda_declared_dim3_user(dim3 d);
  void __cpucuda_dim3_to_arg() {
    dim3 d = __cpucuda_declared_dim3_getter();
    __cpucuda_declared_dim3_user(d);
  }

  void __cpucuda_call_kernel(
      dim3 grid_dim,
      dim3 block_idx,
      dim3 block_dim,
      void** args,
      size_t shared_mem);
  void __cpucuda_call_kernel_self_contained(
      dim3 grid_dim,
      dim3 block_dim,
      void** args,
      size_t shared_mem);
  cudaError_t __cpucudaLaunchKernelSelfContained(
      const void* func,
      dim3 grid_dim,
      dim3 block_dim,
      void** args,
      size_t shared_mem,
      cudaStream_t stream);
  cudaError_t __cpucudaLaunchKernel(
      const void* func,
      dim3 grid_dim,
      dim3 block_dim,
      void** args,
      size_t shared_mem,
      cudaStream_t stream);
  cudaError_t __cpucudaLaunchKernelSelfContainedWithPushedConfiguration(
      const void* func,
      void** args);
  cudaError_t __cpucudaLaunchKernelWithPushedConfiguration(
      const void* func,
      void** args);
  void __cpucuda_declared_function_user() {
    __cpucudaLaunchKernelWithPushedConfiguration(0, 0);
    __cpucudaLaunchKernel(0, {0,0,0}, {0,0,0}, 0, 0, 0);
    __cpucuda_call_kernel({0,0,0}, {0,0,0}, {0,0,0}, 0, 0);

    __cpucudaLaunchKernelSelfContainedWithPushedConfiguration(0, 0);
    __cpucudaLaunchKernelSelfContained(0, {0,0,0}, {0,0,0}, 0, 0, 0);
    __cpucuda_call_kernel_self_contained({0,0,0}, {0,0,0}, 0, 0);
  }

#ifdef __cplusplus
}
#endif

#endif
