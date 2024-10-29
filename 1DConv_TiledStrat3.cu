#include<iostream>
#define SIZE 50
#define FILTER_RADIUS 2
#define TILE_SIZE 16
#define FILTER_WIDTH 2*FILTER_RADIUS + 1
__constant__ float F_d[2*FILTER_RADIUS+1];

__global__ void conv1d_Basic(float* N,float* P){
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    __shared__ float N_s[TILE_SIZE];
    N_s[threadIdx.x] = N[i];
    __syncthreads();
    int this_tile_start = blockIdx.x*blockDim.x;
    int next_tile_start = (blockIdx.x+1)*blockDim.x;
    int start = i - FILTER_RADIUS;
    float pValue = 0;
    for(int j =0;j<FILTER_WIDTH;j++){
        int N_index = start + j;
        if(N_index>=0 && N_index<SIZE){
            if(N_index>= this_tile_start && N_index<next_tile_start){
                pValue +=N_s[threadIdx.x - FILTER_RADIUS +j] * F_d[j];
            }
            else{
                pValue +=N[N_index] * F_d[j];
            }

        }
    }

    P[i] = pValue;
    
}

int main(){
    float* N,*P,*F;
    N = (float*)malloc(SIZE*sizeof(float));
    P = (float*)malloc(SIZE* sizeof(float));
    F = (float*)malloc((2*FILTER_RADIUS+1)*sizeof(float));
    float *N_d,*P_d;
    for(int i=0;i<SIZE;i++){
        N[i] = rand()%10;
    }
    for(int i=0;i<2*FILTER_RADIUS+1;i++)
        F[i] = rand()%10;
    cudaMalloc((void**)&N_d,SIZE*sizeof(float));
    cudaMalloc((void**)&P_d,SIZE*sizeof(float));
    cudaMemcpy(N_d,N,SIZE*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(F_d,F,(2*FILTER_RADIUS+1)*sizeof(float));
    dim3 dimGrid(ceil((float)SIZE/(float)TILE_SIZE),1,1);
    dim3 dimBlock(TILE_SIZE,1,1);
    conv1d_Basic<<<dimGrid,dimBlock>>>(N_d,P_d);
    cudaMemcpy(P,P_d,SIZE*sizeof(float),cudaMemcpyDeviceToHost);
    std::cout<<"Input : \n";
    for(int i=0;i<SIZE;i++)
        std::cout<<N[i]<<" ";
    std::cout<<"\nFilter: \n";
    for(int i=0;i<2*FILTER_RADIUS+1;i++)
        std::cout<<F[i]<<" ";
    std::cout<<"\nOutput : \n";
    for(int i=0;i<SIZE;i++)
        std::cout<<P[i]<<" ";
    free(N);
    free(P);
    free(F);
    cudaFree(N_d);
    cudaFree(P_d);
    return 0;
}