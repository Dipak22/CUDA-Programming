#include <iostream>
#include <string>
#include <cuda_runtime.h>

#define NUM_BINS 7

__global__ void hist_kernel(char *data, int length, unsigned int *hist) {
    __shared__ unsigned int hist_s[NUM_BINS];

    // Initialize shared memory histogram bins
    if (threadIdx.x < NUM_BINS) {
        hist_s[threadIdx.x] = 0;
    }
    __syncthreads();

    // Calculate the index for each thread
    unsigned int accumulator = 0;
    int prevBin = -1;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    for(unsigned int i = index;i<length;i += blockDim.x*gridDim.x){
        int ch = data[i] - 'a';
        if (ch >= 0 && ch < 26) {
            int bin  = ch/4;
            if(bin == prevBin){
                accumulator++;
            }else{
                if(accumulator>0){
                    atomicAdd(&(hist_s[prevBin]), 1);
                }
                prevBin = bin;
                accumulator = 1;
            }
           
        }
    }
    if(accumulator>0){
        atomicAdd(&(hist_s[prevBin]), accumulator);
    }
    __syncthreads();

    // Copy results from shared memory to global memory
    if (threadIdx.x < NUM_BINS && hist_s[threadIdx.x] > 0) {
        atomicAdd(&(hist[threadIdx.x]), hist_s[threadIdx.x]);
    }
}

int main() {
    std::string data = "programming massively parallel processors";
    int length = data.length();

    // Allocate memory for histogram on the host
    unsigned int *hist;
    hist = (unsigned int*)malloc(NUM_BINS * sizeof(int));
    memset(hist, 0, NUM_BINS * sizeof(int));

    // Allocate memory on the device
    char *data_d;
    unsigned int *hist_d;
    cudaMalloc((void**)&data_d, length * sizeof(char));
    cudaMalloc((void**)&hist_d, NUM_BINS * sizeof(int));

    // Copy data to device
    cudaMemcpy(data_d, data.c_str(), length * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemset(hist_d, 0, NUM_BINS * sizeof(int));

    // Define kernel launch parameters
    dim3 dimBlock(8, 1, 1);
    dim3 dimGrid((length + dimBlock.x - 1) / dimBlock.x, 1, 1);

    // Launch kernel
    hist_kernel<<<dimGrid, dimBlock>>>(data_d, length, hist_d);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(hist, hist_d, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the histogram
    std::cout << "Histogram:\n";
    for (int i = 0; i < NUM_BINS; i++) {
        std::cout << hist[i] << " ";
    }

    // Free memory
    free(hist);
    cudaFree(data_d);
    cudaFree(hist_d);
    
    return 0;
}
