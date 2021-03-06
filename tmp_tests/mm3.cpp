
// delarations below would be in some header
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

#define __global__ __attribute__((cpucuda_global))
#define __shared__ __attribute__((cpucuda_shared))
//#define __shared__ __attribute__((annotate("cpucuda_shared")))

// source from here

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

__global__ void mat_mul(float *A, float *B, float *C, int size_x, int size_y)
{
	// these point to the first element of the first block we are considering
	float *a = A + blockIdx.y * BLOCK_SIZE * size_x;
	float *b = B + blockIdx.x * BLOCK_SIZE;
	float *c = C + blockIdx.y * BLOCK_SIZE * size_y + blockIdx.x * BLOCK_SIZE;


	int numBlocks = size_x / BLOCK_SIZE;

	float res;

	for (int i = 0; i < numBlocks; a += BLOCK_SIZE, b += BLOCK_SIZE * size_y, i++) {

		// now a and b point to the first element of the block we are considering

		__shared__ float sa[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float sb[BLOCK_SIZE][BLOCK_SIZE];

		sa[threadIdx.y][threadIdx.x] = a[size_x * threadIdx.y + threadIdx.x];
		sb[threadIdx.y][threadIdx.x] = b[size_y * threadIdx.y + threadIdx.x];

		__syncthreads();

#pragma unroll
		for (int j = 0; j < BLOCK_SIZE; j++) {
			res += sa[threadIdx.y][j] * sb[j][threadIdx.x];
		}
		__syncthreads();
	}
	c[threadIdx.x + threadIdx.y * size_y] = res;
}
