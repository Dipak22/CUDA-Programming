#include<iostream>
#define RADIUS 2
#define SIZE 100
#define TILE_SIZE 16
#define MASK_WIDTH (2*RADIUS+1)
#define INPUT_TILE_SIZE (TILE_SIZE+MASK_WIDTH-1)
__constant__ float M[MASK_WIDTH];

__global__ void conv1d_Tiled(float *N,float *P){
    int tx = threadIdx.x;
    int index = blockIdx.x*TILE_SIZE+threadIdx.x - RADIUS;
    __shared__ float N_s[INPUT_TILE_SIZE];
    if(index>=0 && index<SIZE){
        N_s[tx] = N[index];
    }else
        N_s[tx] = 0.0f;

    __syncthreads();
    int inRow = tx-RADIUS;
    if(index>=0 && index<SIZE && inRow>=0 && inRow<TILE_SIZE){
        float pValue = 0.0f;
        for(int j=0;j<MASK_WIDTH;j++){
            pValue += M[j] * N_s[inRow+j];
        }
        P[index] = pValue;
    }
}

int main(){
    float *N,*P,*M_h;
    N =(float*)malloc(SIZE*sizeof(float));
    P = (float*)malloc(SIZE*sizeof(float));
    M_h = (float*)malloc(MASK_WIDTH*sizeof(float));
    for(int i=0;i<SIZE;i++)
        N[i] = rand()%10;
    for(int i=0;i<MASK_WIDTH;i++)
        M_h[i] = rand()%10;
    cudaMemcpyToSymbol(M,M_h,MASK_WIDTH*sizeof(float));
    float *N_d,*P_d;
    cudaMalloc((void**)&N_d,SIZE*sizeof(float));
    cudaMalloc((void**)&P_d,SIZE*sizeof(float));
    cudaMemcpy(N_d,N,SIZE*sizeof(float),cudaMemcpyHostToDevice);
    dim3 dimBlock(INPUT_TILE_SIZE,1,1);
    dim3 dimGrid((SIZE+TILE_SIZE-1)/TILE_SIZE,1,1);
    conv1d_Tiled<<<dimGrid,dimBlock>>>(N_d,P_d);
    cudaMemcpy(P,P_d,SIZE*sizeof(float),cudaMemcpyDeviceToHost);
    std::cout<<"Input:\n";
    for(int i=0;i<SIZE;i++)
        std::cout<<N[i]<<" ";
    std::cout<<"\nMask:\n";
    for(int i=0;i<MASK_WIDTH;i++)
        std::cout<<M_h[i]<<" ";
    std::cout<<"\nOutput:\n";
    for(int i=0;i<SIZE;i++)
        std::cout<<P[i]<<" ";

    free(N);
    free(P);
    free(M_h);
    cudaFree(N_d);
    cudaFree(P_d);
    return 0;
}