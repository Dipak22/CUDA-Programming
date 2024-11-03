#include <iostream>
#include <cuda_runtime.h>
#define LENGTH 256
#define BLOCKDIM (LENGTH / 2)

__global__ void sum_kernel(int *input, int *output) {
    int index = threadIdx.x;
    __shared__ int input_s[BLOCKDIM];
    input_s[index] = 0;
    // Load data into shared memory with boundary check
    if (index < BLOCKDIM) {
        input_s[index] = input[index] + input[index + BLOCKDIM];
    }
    __syncthreads();

    // Perform reduction within shared memory
    for (unsigned int stride = BLOCKDIM / 2; stride > 0; stride >>=1) {
        if (index < stride) {
           // printf("%d ",index);
            input_s[index] += input_s[index + stride];
        }
        //printf("\n");
        __syncthreads();  // Ensure all threads have completed current stride
    }

    // Write the result to output
    if (index == 0) {
        *output = input_s[0];
    }
}

int main() {
    int *input, *sum;
    input = (int *)malloc(LENGTH * sizeof(int));
    sum = (int *)malloc(sizeof(int));

    // Initialize input data
    for (int i = 0; i < LENGTH; i++) {
        input[i] = rand() % 10;
    }

    // Allocate memory on the device
    int *input_d, *output;
    cudaMalloc((void **)&input_d, LENGTH * sizeof(int));
    cudaMalloc((void **)&output, sizeof(int));

    // Copy input data to the device
    cudaMemcpy(input_d, input, LENGTH * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel with BLOCKDIM threads
    dim3 dimBlock(BLOCKDIM, 1, 1);
    dim3 dimGrid(1, 1, 1);
    sum_kernel<<<dimGrid, dimBlock>>>(input_d, output);

    // Copy the result back to the host
    cudaMemcpy(sum, output, sizeof(int), cudaMemcpyDeviceToHost);

    // Display the input array and the GPU-computed sum
    std::cout << "Input:\n";
    for (int i = 0; i < LENGTH; i++) {
        std::cout << input[i] << " ";
    }
    std::cout << "\nGPU Sum: " << *sum;

    // Calculate and display the CPU sum for verification
    int cpu_sum = 0;
    for (int i = 0; i < LENGTH; i++) {
        cpu_sum += input[i];
    }
    std::cout << "\nCPU Sum: " << cpu_sum << "\n";

    // Free memory
    free(input);
    free(sum);
    cudaFree(input_d);
    cudaFree(output);

    return 0;
}
