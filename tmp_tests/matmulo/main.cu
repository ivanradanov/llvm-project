#include <iostream>
#include <iomanip>
#include <random>
#include <chrono>
#include <assert.h>
#include <stdlib.h>
#include <cstdlib>
#include <omp.h>

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

  std::cout << "Executing warmup" << std::endl;

  // warmup
  {
    for (int i = 0; i < NITERATIONS; ++i) {
      mat_mul<<<grid, block>>>(A, B, C, dimsA.x, dimsB.x);
    }
  }

  std::cout << "Executing kernel" << std::endl;

  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < NITERATIONS; ++i) {
	  mat_mul<<<grid, block>>>(A, B, C, dimsA.x, dimsB.x);
  }

  auto end = std::chrono::high_resolution_clock::now();

  using namespace std::literals;
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  double ms = us / 1000.0;
  double s = ms / 1000.0;
  std::cout << "Executed " << NITERATIONS << " iterations in " << us << "??s ??? " << ms << "ms ??? " << s << "s.\n";


  double matrix_flops = 2.0 * (double) dimsA.x * (double) dimsA.y * (double) dimsB.x;
  double giga = (double) 1000.0 * 1000.0 * 1000.0;
  double gflops = (NITERATIONS * matrix_flops / giga)  / s;

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

  us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  ms = us / 1000.0;
  s = ms / 1000.0;
  std::cout << "Verification mat mul completed in  " << us << "??s ??? " << ms << "ms ??? " << s << "s.\n";

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

void test_cpu_mat_mul();

int main(int argc, char **argv) {

	int block_size = BLOCK_SIZE;

	if (argc != 1 && argc != 5 && argc != 2) {
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
	if (argc == 2) {
		test_cpu_mat_mul();
		return 1;
	}

	dim3 dimsA(k, m, 1);
	dim3 dimsB(n, k, 1);

	run(block_size, dimsA, dimsB);
	return 0;

}

void test_cpu_mat_mul() {
	float *A = (float *) aligned_malloc(default_alignment, sizeof(float) * 1000);
	float *B = (float *) aligned_malloc(default_alignment, sizeof(float) * 1000);
	float *C = (float *) aligned_malloc(default_alignment, sizeof(float) * 1000);
	int i = 0;
	A[i++] = 1;
	A[i++] = 2;
	A[i++] = 3;
	A[i++] = 4;
	A[i++] = 5;
	A[i++] = 6;
	A[i++] = 4;
	A[i++] = 5;
	A[i++] = 6;
	A[i++] = 4;
	A[i++] = 5;
	A[i++] = 6;

	i = 0;
	B[i++] = 7;
	B[i++] = 8;
	B[i++] = 9;
	B[i++] = 10;
	B[i++] = 11;
	B[i++] = 12;

	cpu_mat_mul(A, B, C, 4, 2, 3);

	std::cout << "A" << std::endl;
	print_mat(A, 4, 3);
	std::cout << "B" << std::endl;
	print_mat(B, 3, 2);
	std::cout << "C" << std::endl;
	print_mat(C, 4, 2);

	return;
}
