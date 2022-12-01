#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define N 10000000
#define MAX_ERR 1e-6

__global__ void vector_add(float* out, float* a, float* b, int n) {
  for (int i = 0; i < n; i++) {
    out[i] = a[i] + b[i];
  }
}

int main() {
  float *a, *b, *out;

  // Allocate host memory
  cudaMallocManaged((void**)&a, sizeof(float) * N);
  cudaMallocManaged((void**)&b, sizeof(float) * N);
  cudaMallocManaged((void**)&out, sizeof(float) * N);

  // Initialize host arrays
  for (int i = 0; i < N; i++) {
    a[i] = 1.0f;
    b[i] = 2.0f;
  }

  // Executing kernel
  vector_add<<<1, 1>>>(out, a, b, N);

  cudaDeviceSynchronize();

  // Verification
  for (int i = 0; i < N; i++) {
    assert(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
  }
  printf("out[0] = %f\n", out[0]);
  printf("PASSED\n");

  // Deallocate device memory
  cudaFree(a);
  cudaFree(b);
  cudaFree(out);
}