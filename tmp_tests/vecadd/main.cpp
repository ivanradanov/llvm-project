#include <iostream>
#include <iomanip>
#include <random>
#include <chrono>
#include <assert.h>
#include <stdlib.h>

int NITERATIONS = 1;

#include "__cpucuda_internal_header.h"

__global__ void vec_add(float *A, float *B, float *C, int size)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < size)
		C[i] = A[i] + B[i];
}

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

bool array_equal(float *a, float *b, int size) {
	double e = 1.e-6;  // epsilon
	for (int i = 0; i < size; ++i)
		if (fabs(a[i] - b[i]) > e)
			return false;
	return true;
}

void cpu_vec_add(float *A, float *B, float *C, int size) {
  for (int m = 0; m < size; m++)
	  C[m] = A[m] + B[m];
}

void print_vec(float *A, int size) {
	for (int m = 0; m < size; m++)
		std::cout << std::fixed << std::setprecision(2) << A[m] << " ";
	std::cout << std::endl;
}

void run(int block_size, int size) {

	std::cout << "size " << size << std::endl;

	float *A = (float *) aligned_malloc(default_alignment, sizeof(float) * size);
	float *B = (float *) aligned_malloc(default_alignment, sizeof(float) * size);

	// Random floats 0.0 - 1.0
	populate_array(A, size);
	populate_array(B, size);

	float *C = (float *) aligned_malloc(default_alignment, sizeof(float) * size);

  std::cout << "Executing kernel" << std::endl;

  auto start = std::chrono::high_resolution_clock::now();

  dim3 grid((size + block_size - 1) / block_size);
  dim3 block(block_size);

  // warmup
  {
    __cpucuda_global_gridDim = grid;
    __cpucuda_global_blockDim = block;

    for (int i = 0; i < NITERATIONS; ++i) {
#pragma omp parallel for collapse(3)
      for(size_t g_x = 0; g_x < grid.x; ++g_x){
        for(size_t g_y = 0; g_y < grid.y; ++g_y){
          for(size_t g_z = 0; g_z < grid.z; ++g_z){
            dim3 block_index(g_x, g_y, g_z);
            __cpucuda_global_blockIdx = block_index;
            vec_add(A, B, C, size);
          }
        }
      }
	  }
  }
  __cpucuda_global_gridDim = grid;
  __cpucuda_global_blockDim = block;

  for (int i = 0; i < NITERATIONS; ++i) {
#pragma omp parallel for collapse(3)
    for(size_t g_x = 0; g_x < grid.x; ++g_x){
      for(size_t g_y = 0; g_y < grid.y; ++g_y){
        for(size_t g_z = 0; g_z < grid.z; ++g_z){
	        dim3 block_index(g_x, g_y, g_z);
	        __cpucuda_global_blockIdx = block_index;
	        vec_add(A, B, C, size);
        }
      }
    }
  }

  auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  using namespace std::literals;
  std::cout << "Executed " << NITERATIONS << " iterations in " << duration.count() << "µs ≈ "
            << (end - start) / 1ms << "ms ≈ "
            << (end - start) / 1s << "s.\n";

  double vec_add_flops = size;
  double gflops = (NITERATIONS * vec_add_flops / (double) (1000.0 * 1000.0 * 1000.0)) /
	  (duration.count() / (double) 1000.0 * 1000.0);

  std::cout << "GFlop/s: " << gflops << std::endl << std::endl;

  std::cout
	  << omp_get_max_threads() << ", "
	  << size << ", "
	  << NITERATIONS << ", "
	  << ms << ", "
	  << gflops << ", "
	  << std::endl
	  << std::endl;


  std::cout << "Running verification..." << std::endl;

  float *C2 = (float *) aligned_malloc(default_alignment, sizeof(float) * size);

  cpu_vec_add(A, B, C2, size);
  /*
  std::cout << "C" << std::endl;
  print_vec(C, size);
  std::cout << "C2" << std::endl;
  print_vec(C2, size);
  */

  if (array_equal(C, C2, size))
	  std::cout << "PASS" << std::endl;
  else
    std::cout << "FAILED" << std::endl;

}

int main(int argc, char **argv) {

	int block_size = 512;

	if (argc != 1 && argc != 3) {
		std::cout << "Usage: ./a.out <size> <n_iters>" << std::endl;
		return 1;
	}
	int size;
	if (argc == 3) {
		int i = 1;
		size = atoi(argv[i++]);
		NITERATIONS = atoi(argv[i++]);
	} else {
		size = 100;
	}

	run(block_size, size);

}
