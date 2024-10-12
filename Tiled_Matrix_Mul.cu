#include<iostream>
#define M 4
#define N 5
#define P 6
#define TILE_WIDTH 2

__global__ void mat_mul(float* A,float *B,float *C){
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    int row = by*TILE_WIDTH + ty;
    int col = bx*TILE_WIDTH + tx;
    float Pvalue =0;
    for(int ph =0;ph<ceil((float)N/TILE_WIDTH);++ph){
        if(row<M && (ph*TILE_WIDTH + tx)<N)
            Mds[ty][tx] = A[row*N + (ph*TILE_WIDTH + tx)];
        else
            Mds[ty][tx] = 0.0f;
        if((ph*TILE_WIDTH+ty)<N && col<P)
            Nds[ty][tx] = B[(ph*TILE_WIDTH+ty)*P + col];
        else
            Nds[ty][tx] =0.0f;
        __syncthreads();

        for(int k=0;k<TILE_WIDTH;++k){
            Pvalue +=Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }
    if(row<M && col<P)
        C[row*P + col] = Pvalue;
}

int main(){
    float *h_A,*h_B,*h_C;
    h_A = (float*) malloc(M* N * sizeof(float));
    h_B = (float*)malloc(N*P*sizeof(float));
    h_C = (float*)malloc(M*P*sizeof(float));

    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            h_A[i*N+j] = rand()%5;
        }
    }
    for(int i=0;i<N;i++){
        for(int j=0;j<P;j++){
            h_B[i*P+j] = rand()%5;
        }
    }

    float *d_A,*d_B,*d_C;
    cudaMalloc((void**)&d_A,M*N*sizeof(float));
    cudaMalloc((void**)&d_B,N*P*sizeof(float));
    cudaMalloc((void**)&d_C,M*P*sizeof(float));

    cudaMemcpy(d_A,h_A,M*N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,N*P*sizeof(float),cudaMemcpyHostToDevice);
    dim3 dimBlock(TILE_WIDTH,TILE_WIDTH);
    dim3 dimGrid(ceil((float)P/TILE_WIDTH),ceil((float)M/TILE_WIDTH));
    mat_mul<<<dimGrid,dimBlock>>>(d_A,d_B,d_C);

    cudaMemcpy(h_C,d_C,M*P*sizeof(float),cudaMemcpyDeviceToHost);

    std::cout<<"Matrix A:\n";

    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            std::cout<<h_A[i*N+j]<<" ";
        }
        std::cout<<"\n";
    }
    std::cout<<"\n Matrix B: \n";
    for(int i=0;i<N;i++){
        for(int j=0;j<P;j++){
            std::cout<<h_B[i*P+j]<<" ";
        }
        std::cout<<"\n";
    }
    std::cout<<"\n Matrix C:\n";
    for(int i=0;i<M;i++){
        for(int j=0;j<P;j++){
            std::cout<<h_C[i*P+j]<<" ";
        }
        std::cout<<"\n";
    }

    std::cout<<"\nCPU validatated Matrix C \n";
    for(int i =0;i<M;i++){
        for(int j=0;j<P;j++){
            float sum =0.0f;
            for(int k =0;k<N;k++){
                sum +=h_A[i*N+k]*h_B[k*P + j];
            }
            std::cout<<sum<<" ";
        }
        std::cout<<"\n";
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}