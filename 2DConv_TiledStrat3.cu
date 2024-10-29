#include <iostream>
#include <cstdlib>

#define RADIUS 5
#define TILE_SIZE 32
#define WIDTH 200
#define MASK_WIDTH (2 * RADIUS + 1)

__constant__ float M[MASK_WIDTH * MASK_WIDTH];

__global__ void conv2d_tiledCached(float* N, float* P) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Shared memory for the core tile (without halo)
    __shared__ float Ns[TILE_SIZE][TILE_SIZE];

    // Load only the core tile into shared memory (within bounds check)
    if (row < WIDTH && col < WIDTH) {
        Ns[threadIdx.y][threadIdx.x] = N[row * WIDTH + col];
    } else {
        Ns[threadIdx.y][threadIdx.x] = 0.0f; // Padding for out-of-bound access
    }

    __syncthreads(); // Ensure all threads have loaded data into shared memory

    // Perform convolution if within bounds
    if (row < WIDTH && col < WIDTH) {
        float pValue = 0.0f;

        // Apply convolution filter using both shared memory for core and global memory for halo
        for (int r = 0; r < MASK_WIDTH; r++) {
            for (int c = 0; c < MASK_WIDTH; c++) {
                int globalRow = row - RADIUS + r;
                int globalCol = col - RADIUS + c;

                float pixelValue;

                // If globalRow/globalCol is within the image bounds, fetch from either shared or global memory
                if (globalRow >= 0 && globalRow < WIDTH && globalCol >= 0 && globalCol < WIDTH) {
                    if (threadIdx.y - RADIUS + r >= 0 && threadIdx.y - RADIUS + r < TILE_SIZE &&
                        threadIdx.x - RADIUS + c >= 0 && threadIdx.x - RADIUS + c < TILE_SIZE) {
                        // Fetch from shared memory if within the core tile
                        pixelValue = Ns[threadIdx.y - RADIUS + r][threadIdx.x - RADIUS + c];
                    } else {
                        // Fetch from global memory (halo region)
                        pixelValue = N[globalRow * WIDTH + globalCol];
                    }
                } else {
                    // Out-of-bounds values are treated as 0 (zero padding)
                    pixelValue = 0.0f;
                }

                // Apply the convolution mask
                pValue += M[r * MASK_WIDTH + c] * pixelValue;
            }
        }

        // Store the result
        P[row * WIDTH + col] = pValue;
    }
}

int main() {
    float *N, *P, *M_h;
    N = (float*)malloc(WIDTH * WIDTH * sizeof(float));
    P = (float*)malloc(WIDTH * WIDTH * sizeof(float));
    M_h = (float*)malloc(MASK_WIDTH * MASK_WIDTH * sizeof(float));

    // Initialize the input matrix N and the mask M_h
    for (int i = 0; i < WIDTH; i++) {
        for (int j = 0; j < WIDTH; j++) {
            N[i * WIDTH + j] = rand() % 10;
        }
    }
    for (int i = 0; i < MASK_WIDTH; i++) {
        for (int j = 0; j < MASK_WIDTH; j++) {
            M_h[i * MASK_WIDTH + j] = rand() % 10;
        }
    }

    // Copy the kernel to constant memory
    cudaMemcpyToSymbol(M, M_h, MASK_WIDTH * MASK_WIDTH * sizeof(float));

    // Allocate memory on the GPU
    float *N_d, *P_d;
    cudaMalloc((void**)&N_d, WIDTH * WIDTH * sizeof(float));
    cudaMalloc((void**)&P_d, WIDTH * WIDTH * sizeof(float));

    // Copy the input data to the GPU
    cudaMemcpy(N_d, N, WIDTH * WIDTH * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 dimBlock(TILE_SIZE, TILE_SIZE, 1);
    dim3 dimGrid((WIDTH + TILE_SIZE - 1) / TILE_SIZE, (WIDTH + TILE_SIZE - 1) / TILE_SIZE, 1);

    // Launch the kernel
    conv2d_tiledCached<<<dimGrid, dimBlock>>>(N_d, P_d);

    // Error checking after kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy the result back to the host
    cudaMemcpy(P, P_d, WIDTH * WIDTH * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the input matrix
    std::cout << "Input: \n";
    for (int i = 0; i < WIDTH; i++) {
        for (int j = 0; j < WIDTH; j++) {
            std::cout << N[i * WIDTH + j] << " ";
        }
        std::cout << "\n";
    }

    // Print the mask
    std::cout << "\nMask:\n";
    for (int i = 0; i < MASK_WIDTH; i++) {
        for (int j = 0; j < MASK_WIDTH; j++) {
            std::cout << M_h[i * MASK_WIDTH + j] << " ";
        }
        std::cout << "\n";
    }

    // Print the output matrix
    std::cout << "\nOutput:\n";
    for (int i = 0; i < WIDTH; i++) {
        for (int j = 0; j < WIDTH; j++) {
            std::cout << P[i * WIDTH + j] << " ";
        }
        std::cout << "\n";
    }

    // Free the allocated memory
    free(N);
    free(P);
    free(M_h);

    cudaFree(N_d);
    cudaFree(P_d);

    return 0;
}
