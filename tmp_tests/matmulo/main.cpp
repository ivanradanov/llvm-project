#include <iostream>
#include <iomanip>
#include <random>
#include <chrono>
#include <assert.h>
#include <stdlib.h>
#include <cstdlib>
#include <omp.h>

#include "__cpucuda_internal_header.h"

int NITERATIONS = 1;

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

	float res = 0;

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

void cpu_mat_mul(float *A, float *B, float *C, int size_m, int size_n, int size_k) {
  for (int m = 0; m < size_m; m++) {
    for (int n = 0; n < size_n; n++) {
	    int Ci = n + m * size_n;
	    C[Ci] = 0;
      for (int k = 0; k < size_k; k++) {
	      int Ai = k + m * size_k;
	      int Bi = n + k * size_n;
	      C[Ci] += A[Ai] * B[Bi];
      }
    }
  }
}

void print_mat(float *A, int size_m, int size_n) {
	for (int m = 0; m < size_m; m++) {
		for (int n = 0; n < size_n; n++) {
			int Ai = n + m * size_n;
			std::cout << std::fixed << std::setprecision(2) << A[Ai] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

void run(int block_size, dim3 dimsA, dim3 dimsB) {

	std::cout << "dimsA "
	          << dimsA.x << " "
	          << dimsA.y << " "
	          << dimsA.z << std::endl;
	std::cout << "dimsB "
	          << dimsB.x << " "
	          << dimsB.y << " "
	          << dimsB.z << std::endl;

	float *A = (float *) aligned_malloc(default_alignment, sizeof(float) * dimsA.x * dimsA.y);
	float *B = (float *) aligned_malloc(default_alignment, sizeof(float) * dimsB.x * dimsB.y);

	// Random floats 0.0 - 1.0
	populate_array(A, dimsA.x * dimsA.y);
	populate_array(B, dimsB.x * dimsB.y);

	dim3 dimsC(dimsB.x, dimsA.y, 1);
	float *C = (float *) aligned_malloc(default_alignment, sizeof(float) * dimsC.x * dimsC.y);

	dim3 block(block_size, block_size);
  dim3 grid(dimsB.x / block.x, dimsA.y / block.y);

  std::cout << "Executing kernel" << std::endl;

  /*
  std::cout << "A" << std::endl;
  print_mat(A, dimsA.x, dimsA.y);
  std::cout << "B" << std::endl;
  print_mat(B, dimsB.x, dimsB.y);
  std::cout << "C" << std::endl;
  print_mat(C, dimsC.x, dimsC.y);
  */


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
            mat_mul(A, B, C, dimsA.x, dimsB.x);
          }
        }
      }
	  }
  }

  auto start = std::chrono::high_resolution_clock::now();

  __cpucuda_global_gridDim = grid;
  __cpucuda_global_blockDim = block;

  for (int i = 0; i < NITERATIONS; ++i) {
#pragma omp parallel for collapse(3)
    for(size_t g_x = 0; g_x < grid.x; ++g_x){
      for(size_t g_y = 0; g_y < grid.y; ++g_y){
        for(size_t g_z = 0; g_z < grid.z; ++g_z){
	        dim3 block_index(g_x, g_y, g_z);
	        __cpucuda_global_blockIdx = block_index;
	        mat_mul(A, B, C, dimsA.x, dimsB.x);
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

  auto ms = (end - start) / 1ms;

  double matrix_flops = 2.0 * (double) dimsA.x * (double) dimsA.y * (double) dimsB.x;
  double gflops = (NITERATIONS * matrix_flops / (double) 1000000000.0)  / ((end - start).count() / (double) 1000000.0);

  std::cout << "GFlop/s: " << gflops << std::endl << std::endl;

  std::cout
	  << omp_get_max_threads() << ", "
	  << dimsA.y << ", "
	  << dimsB.x << ", "
	  << dimsA.x << ", "
	  << NITERATIONS << ", "
	  << ms << ", "
	  << gflops << ", "
	  << std::endl
    << std::endl;


  std::cout << "Running verification..." << std::endl;

  float *C2 = (float *) aligned_malloc(default_alignment, sizeof(float) * dimsC.x * dimsC.y);

  assert(dimsB.y == dimsA.x);

  start = std::chrono::high_resolution_clock::now();
  cpu_mat_mul(A, B, C2, dimsA.y, dimsB.x, dimsA.x);
  end = std::chrono::high_resolution_clock::now();

  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  using namespace std::literals;
  std::cout << "Verification mat mul completed in  " << duration.count() << "µs ≈ "
            << (end - start) / 1ms << "ms ≈ "
            << (end - start) / 1s << "s.\n";


  auto success = array_equal(C, C2, dimsC.x * dimsC.y);

  if (success)
	  std::cout << "PASS" << std::endl;
  else
    std::cout << "FAILED" << std::endl;


  /*
    std::cout << "A" << std::endl;
    print_mat(A, dimsA.x, dimsA.y);
    std::cout << "B" << std::endl;
    print_mat(B, dimsB.x, dimsB.y);
    std::cout << "C" << std::endl;
    print_mat(C, dimsC.x, dimsC.y);
    std::cout << "C2" << std::endl;
    print_mat(C2, dimsC.x, dimsC.y);
  */


}

int main(int argc, char **argv) {

	int block_size = BLOCK_SIZE;

	if (argc != 1 && argc != 5) {
		std::cout << "Usage: ./a.out <m> <n> <k> <n_iters>" << std::endl;
		return 1;
	}
	int m, n, k;
	if (argc == 5) {
    int i = 1;
    m = atoi(argv[i++]);
    n = atoi(argv[i++]);
    k = atoi(argv[i++]);
    assert(m % block_size == 0);
    assert(n % block_size == 0);
    assert(k % block_size == 0);
    NITERATIONS = atoi(argv[i++]);
	} else {
		m = 2 * block_size;
		n = 3 * block_size;
		k = 4 * block_size;
	}

	dim3 dimsA(k, m, 1);
	dim3 dimsB(n, k, 1);

	run(block_size, dimsA, dimsB);

}
