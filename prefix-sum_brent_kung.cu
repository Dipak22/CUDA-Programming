#include <iostream>
#include <cuda_runtime.h>

#define LENGTH 1024
#define SEGMENT 1024
#define BLOCK_DIM (SEGMENT / 2)

__global__ void prefix_sum(int *input, int *output) {
    __shared__ int T[SEGMENT];
    
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x * 2 + tid;
    
    // Load data from global memory to shared memory
    T[tid] = (index < LENGTH) ? input[index] : 0;
    T[tid + BLOCK_DIM] = (index + BLOCK_DIM < LENGTH) ? input[index + BLOCK_DIM] : 0;
    __syncthreads();

    int stride = 1;
    // Up-sweep (reduction) phase
    while (stride < SEGMENT) {
        int idx = (tid + 1) * stride * 2 - 1;
        if (idx < SEGMENT && (idx - stride) >= 0) {
            T[idx] += T[idx - stride];
        }
        stride *= 2;
        __syncthreads();
    }

    // Down-sweep phase
    stride = BLOCK_DIM;
    while (stride > 0) {
        __syncthreads();
        int idx = (tid + 1) * stride * 2 - 1;
        if (idx + stride < SEGMENT) {
            T[idx + stride] += T[idx];
        }
        stride >>= 1;
    }
    __syncthreads();

    // Write results back to global memory
    if (index < LENGTH) {
        output[index] = T[tid];
    }
    if (index + BLOCK_DIM < LENGTH) {
        output[index + BLOCK_DIM] = T[tid + BLOCK_DIM];
    }
}

int main() {
    int *input, *output;
    input = (int*)malloc(LENGTH * sizeof(int));
    output = (int*)malloc(LENGTH * sizeof(int));
    
    for (int i = 0; i < LENGTH; i++) {
        input[i] = rand() % 10;
    }
    
    int *input_d, *output_d;
    cudaMalloc((void**)&input_d, LENGTH * sizeof(int));
    cudaMalloc((void**)&output_d, LENGTH * sizeof(int));
    
    cudaMemcpy(input_d, input, LENGTH * sizeof(int), cudaMemcpyHostToDevice);
    
    dim3 dimBlock(BLOCK_DIM, 1, 1);
    dim3 dimGrid((LENGTH + SEGMENT - 1) / SEGMENT, 1, 1);
    
    prefix_sum<<<dimGrid, dimBlock>>>(input_d, output_d);
    cudaDeviceSynchronize();

    cudaMemcpy(output, output_d, LENGTH * sizeof(int), cudaMemcpyDeviceToHost);
    
    std::cout << "gpu input\n";
    for (int i = 0; i < LENGTH; i++) {
        std::cout << input[i] << " ";
    }
    std::cout << "\ngpu output\n";
    for (int i = 0; i < LENGTH; i++) {
        std::cout << output[i] << " ";
    }
    
    free(input);
    free(output);
    cudaFree(input_d);
    cudaFree(output_d);
    
    return 0;
}
