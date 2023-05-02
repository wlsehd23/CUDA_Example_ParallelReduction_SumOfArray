#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#define N (256 * 256)
#define THREADS_PER_BLOCK 512
#include <chrono> // To measure time with high precision
using namespace std;


cudaError_t CUDA_Sum(int* h_y, int* h_X, unsigned int size);
cudaError_t CUDA_Atomic_Sum(int* h_y, int* h_X, unsigned int size);

__global__ void CUDA_Sum_kernel(int* d_y, int* d_X)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    d_y[0] += d_X[index];

}

__global__ void CUDA_Atomic_Sum_kernel(int* d_y, int* d_X)
{
    __shared__ int temp[THREADS_PER_BLOCK];
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    temp[threadIdx.x] = d_X[index];

    __syncthreads();

    if (threadIdx.x == 0) {
        int sum = 0;
        for (int i = 0; i < THREADS_PER_BLOCK; i++) {
            sum += temp[i];
        }
        atomicAdd(d_y, sum);
    }
}


int main()
{
    printf("Hello?\n");

    int* h_X = new int[N];
    int* h_y = new int[1];

    // CPU
    h_y[0] = 0;

    chrono::system_clock::time_point ct0, ct1;
    chrono::microseconds cDt1;
    ct0 = chrono::system_clock::now();
    for (int i = 0; i < N; i++)
    {
        h_X[i] = i%256;
        h_y[0] += h_X[i];
    }
    ct1 = chrono::system_clock::now();
    cDt1 = chrono::duration_cast<chrono::microseconds>(ct1 - ct0);
    printf("CPU Sum = %d [ET=%d usec]\n", h_y[0], cDt1.count());

    // GPU wp Atomic
    h_y[0] = 0;
    CUDA_Sum(h_y, h_X, N);

    h_y[0] = 0;
    CUDA_Atomic_Sum(h_y, h_X, N);

    cudaThreadSynchronize();

    return 0;

}


// Helper function for using CUDA to add vectors in parallel.
cudaError_t CUDA_Sum(int* h_y, int* h_X, unsigned int size)
{
    int* d_X, * d_y;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    // Allocate GPU buffers for decive vectors.
    cudaMalloc(&d_X, size * sizeof(int));
    cudaMalloc(&d_y, 1 * sizeof(int));
    cudaMemcpy(d_X, h_X, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, 1 * sizeof(int), cudaMemcpyHostToDevice);
    // Launch a kernel on the GPU with one thread for each element.
    int BLOCK_SIZE = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    // Execute
    chrono::system_clock::time_point ct0, ct1;
    chrono::microseconds cDt1;
    ct0 = chrono::system_clock::now();

    CUDA_Sum_kernel << <BLOCK_SIZE, THREADS_PER_BLOCK >> > (d_y, d_X);

    ct1 = chrono::system_clock::now();
    cDt1 = chrono::duration_cast<chrono::microseconds>(ct1 - ct0);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();

    cudaMemcpy(h_y, d_y, 1 * sizeof(int), cudaMemcpyDeviceToHost);

    // print result
    printf("GPU Sum w/o atomicAdd= %d [ET=%d usec]\n", h_y[0], cDt1.count());

Error:
    cudaFree(d_y);
    cudaFree(d_X);

    return cudaStatus;
}


cudaError_t CUDA_Atomic_Sum(int* h_y, int* h_X, unsigned int size) 
{
    int* d_X, * d_y;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    // Allocate GPU buffers for decive vectors.
    cudaMalloc(&d_X, size * sizeof(int));
    cudaMalloc(&d_y, 1 * sizeof(int));
    cudaMemcpy(d_X, h_X, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, 1 * sizeof(int), cudaMemcpyHostToDevice);
    // Launch a kernel on the GPU with one thread for each element.
    int BLOCK_SIZE = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    // Execute
    chrono::system_clock::time_point ct0, ct1;
    chrono::microseconds cDt1;
    ct0 = chrono::system_clock::now();

    CUDA_Atomic_Sum_kernel << <BLOCK_SIZE, THREADS_PER_BLOCK >> > (d_y, d_X);

    ct1 = chrono::system_clock::now();
    cDt1 = chrono::duration_cast<chrono::microseconds>(ct1 - ct0);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();

    cudaMemcpy(h_y, d_y, 1 * sizeof(int), cudaMemcpyDeviceToHost);

    // print result
    printf("GPU Sum w/ atomicAdd= %d [ET=%d usec]\n", h_y[0], cDt1.count());

Error:
    cudaFree(d_y);
    cudaFree(d_X);

    return cudaStatus;
}