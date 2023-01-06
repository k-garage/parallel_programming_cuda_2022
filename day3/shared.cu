#include <bits/stdc++.h>
#include <cuda_runtime.h>

__global__ void hi(int* a) {
  // register (local variable)
  int x = 0;
  int idx = threadIdx.x;

  // shared memory
  __shared__ int s_a[64];
  // assign value to shared memory
  for (int i = 0; i < 4; i++) {
    x = idx + i;
    s_a[idx] = x * idx;
  }
  // synchronize
  __syncthreads();
  // access shared memory
  int sum = 0;
  for (int i = 0; i < 64; i++) sum += s_a[i];

  // assign value to global memory
  a[idx] = sum + idx;
}

int main() {
  int* a;
  cudaMallocManaged(&a, 1024 * sizeof(int));

  hi<<<1, 64>>>(a);
  cudaDeviceSynchronize();

  for (int i = 0; i < 64; i++) printf("%d\n", a[i]);

  cudaFree(a);

  return 0;
}