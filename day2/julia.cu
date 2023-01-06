/*
# Before compile, install opencv2 with:

sudo apt-get install libopencv-dev

* Compile : nvcc -o fractal fractal.cu -I.. -lcuda $(pkg-config opencv4 --libs
--cflags)
* Run : ./fractal < image file path>
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#define MANDELBROT_ITERATIONS 256
#define JULIA_ITERATIONS 256

// Input image has 3 channels corresponding to RGB
// The input image is encoded as unsigned characters [0, 255]
__global__ void fractal(unsigned char* out, int width, int height) {
  int Row = blockIdx.y * blockDim.y + threadIdx.y;
  int Col = blockIdx.x * blockDim.x + threadIdx.x;

  if (Row < height && Col < width) {
    for (int i = 0; i < MANDELBROT_ITERATIONS; i++) {
      float x = Col / (float)width * 4.0 - 2.0;
      float y = Row / (float)height * 4.0 - 2.0;
      float x0 = x;
      float y0 = y;
      int iteration = 0;
      while (x * x + y * y < 4 && iteration < MANDELBROT_ITERATIONS) {
        float xtemp = x * x - y * y + x0;
        y = 2 * x * y + y0;
        x = xtemp;
        iteration++;
      }
      out[Row * width + Col] = iteration;
    }
  }
}

__global__ void julia(unsigned char* out, int width, int height) {
  int Row = blockIdx.y * blockDim.y + threadIdx.y;
  int Col = blockIdx.x * blockDim.x + threadIdx.x;

  if (Row < height && Col < width) {
    float x = Col / (float)width * 4.0 - 2.0;
    float y = Row / (float)height * 4.0 - 2.0;
    float x0 = x;
    float y0 = y;
    float c_real = -0.8;
    float c_imag = 0.156;
    int iteration = 0;
    while (x * x + y * y < 4 && iteration < JULIA_ITERATIONS) {
      float xtemp = x * x - y * y + c_real;
      y = 2 * x * y + c_imag;
      x = xtemp;
      iteration++;
    }
    out[Row * width + Col] = iteration;
  }
}

void Usage(char prog_name[]) {
  fprintf(stderr, "Usage: %s <image output path>\n", prog_name);
  exit(EXIT_FAILURE);
}

int main(int argc, char** argv) {
  if (argc != 2) {
    Usage(argv[0]);
  }

  const char* file_name = argv[1];
  int width = 512, height = 512;

  unsigned char* h_resultImg;
  unsigned char* d_resultImg;

  h_resultImg = (unsigned char*)malloc(width * height * sizeof(unsigned char));
  cudaMalloc((void**)&d_resultImg, width * height * sizeof(unsigned char));

  // Launch the Kernel
  const int block_size = 16;
  dim3 threads(block_size, block_size);
  dim3 grid(ceil(width / (double)threads.x), ceil(height / (double)threads.y));
  // fractal << <grid, threads >> > (d_resultImg, width, height);
  julia<<<grid, threads>>>(d_resultImg, width, height);

  // Copy the device result in device memory to the host result in host memory
  cudaMemcpy(h_resultImg, d_resultImg, width * height * sizeof(unsigned char),
             cudaMemcpyDeviceToHost);

  cv::Mat resultImg(height, width, CV_8UC1);
  memcpy(resultImg.data, h_resultImg, width * height);

  // Free device global memory
  cudaFree(d_resultImg);

  // Free host memory
  free(h_resultImg);

  // cv::Mat resizeImg;
  cv::resize(resultImg, resultImg, cv::Size(width / 2, height / 2));
  // save image to filename.jpg
  cv::imwrite(file_name, resultImg);

  return 0;
}