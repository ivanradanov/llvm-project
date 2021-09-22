
// TODO the clang CUDA builtin headers actually define the constructer as such:

// dim3(unsigned x, unsigned y = 1, unsigned z = 1) : x(x), y(y), z(z) {}

// however, with this definition, you cannot create a dim3 without supplying an
// argument for x which is not how it works in cuda I believe, is this
// constructor ok?
struct dim3 {
  unsigned x, y, z;
  dim3(unsigned x = 1, unsigned y = 1, unsigned z = 1) : x(x), y(y), z(z) {}
};

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

void __cpucuda_real_func_user() {
	__cpucuda_real_blockIdx();
	__cpucuda_real_blockDim();
	__cpucuda_real_gridDim();
}

#define threadIdx __cpucuda_threadIdx()
#define blockIdx __cpucuda_blockIdx()
#define blockDim __cpucuda_blockDim()
#define gridDim __cpucuda_gridDim()
#define __syncthreads __cpucuda_syncthreads

//#define __global__ __attribute__((cpucuda_global))
//#define __shared__ __attribute__((cpucuda_shared))
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))
//#define __shared__ __attribute__((annotate("cpucuda_shared")))
