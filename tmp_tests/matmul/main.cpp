#include <iostream>
#include <random>
#include <chrono>
#include <assert.h>

#define NITERATIONS 1
#define BLOCK_SIZE 32


// ------------------------------------------------------------------------------------- //
// ------------------------------------------------------------------------------------- //
struct dim3 {
	unsigned x, y, z;
	dim3(unsigned x = 1, unsigned y = 1, unsigned z = 1) : x(x), y(y), z(z) {}
};

// ------------------------------------------------------------------------------------- //
// ------------------------------------------------------------------------------------- //

dim3 __cpucuda_global_blockDim;
dim3 __cpucuda_global_gridDim;
dim3 __cpucuda_global_blockIdx;

dim3 __cpucuda_real_blockDim() {
	return __cpucuda_global_blockDim;
}
dim3 __cpucuda_real_gridDim() {
	return __cpucuda_global_gridDim;
}
dim3 __cpucuda_real_blockIdx() {
	return __cpucuda_global_blockIdx;
}
// ------------------------------------------------------------------------------------- //
// ------------------------------------------------------------------------------------- //

void mat_mul(float *A, float *B, float *C, int size_x, int size_y);


constexpr std::size_t default_alignment = sizeof(double) * 16;

inline void* aligned_malloc(size_t align, size_t size)
{
	assert(align >= sizeof(void*));

	if (size == 0) {
		return nullptr;
	}

	void* result = nullptr;
	int err = posix_memalign(&result, align, size);

	if (err != 0) {
		return nullptr;
	}

	return result;
}

void populate_array(float *a, int size) {
	for (int i = 0; i < size; ++i)
		a[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
}

void run(int block_size, dim3 dimsA, dim3 dimsB) {

	float *A = (float *) aligned_malloc(default_alignment, sizeof(float) * dimsA.x * dimsA.y);
	float *B = (float *) aligned_malloc(default_alignment, sizeof(float) * dimsB.x * dimsB.y);

	// Random floats 0.0 - 1.0
	populate_array(A, dimsA.x * dimsA.y);
	populate_array(B, dimsB.x * dimsB.y);

	dim3 dimsC(dimsB.x, dimsA.y, 1);
	float *C = (float *) aligned_malloc(default_alignment, sizeof(float) * dimsC.x * dimsC.y);

	dim3 block(block_size, block_size);
  dim3 grid(dimsB.x / block.x, dimsA.y / block.y);

  std::cout << "Executing kernels" << std::endl;

  auto start = std::chrono::high_resolution_clock::now();

  __cpucuda_global_gridDim = grid;
  __cpucuda_global_blockDim = block;

  for (int i = 0; i < NITERATIONS; ++i) {
#pragma omp parallel for collapse(3)
    for(size_t g_x = 0; g_x < grid.x; ++g_x){
      for(size_t g_y = 0; g_y < grid.y; ++g_y){
        for(size_t g_z = 0; g_z < grid.z; ++g_z){
	        dim3 blockIdx = dim3(g_x, g_y, g_z);
	        __cpucuda_global_blockIdx = blockIdx;
	        mat_mul(A, B, C, dimsA.x, dimsB.x);
        }
      }
    }
  }

  auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "Executed " << NITERATIONS << " iterations in " << duration.count() << "ms" <<std::endl;

  std::cout << "Running verification..." << std::endl;

}

int main(int argc, char **argv) {


	int block_size = BLOCK_SIZE;

	dim3 dimsA(5 * 2 * block_size, 5 * 2 * block_size, 1);
	dim3 dimsB(5 * 4 * block_size, 5 * 2 * block_size, 1);

	run(block_size, dimsA, dimsB);

}
