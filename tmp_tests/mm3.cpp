
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

struct dim3 {
  unsigned int x, y, z;
};

extern dim3 __cpucuda_threadIdx;
extern dim3 __cpucuda_blockIdx;
extern dim3 __cpucuda_blockDim;
extern dim3 __cpucuda_gridDim;
void __cpucuda_syncthreads();

#define threadIdx __cpucuda_threadIdx
#define blockIdx __cpucuda_blockIdx
#define blockDim __cpucuda_blockDim
#define gridDim __cpucuda_gridDim
#define __syncthreads __cpucuda_syncthreads

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
