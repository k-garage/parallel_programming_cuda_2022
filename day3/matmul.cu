#include <bits/stdc++.h>
#include <cuda_runtime.h>

#define N 1024

__global__ void SimpleMatmul(float* A, float* B, float* C) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < N && col < N) {
    float sum = 0;
    for (int k = 0; k < N; k++) {
      sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
  }
}

int main() {
  float* A;
  float* B;
  float* C;
  cudaMallocManaged(&A, N * N * sizeof(float));
  cudaMallocManaged(&B, N * N * sizeof(float));
  cudaMallocManaged(&C, N * N * sizeof(float));

  // Initialize A and B
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[i * N + j] = i;
      B[i * N + j] = j;
    }
  }

  dim3 block(16, 16);
  dim3 grid(ceil(N / block.x), ceil(N / block.y));
  SimpleMatmul<<<grid, block>>>(A, B, C);

  cudaDeviceSynchronize();

  printf("Output:\n");
  printf("1, 1 = %f\n", C[0]);
  printf("1, 2 = %f\n", C[1]);
  printf("2, 3 = %f\n", C[N + 2]);
}