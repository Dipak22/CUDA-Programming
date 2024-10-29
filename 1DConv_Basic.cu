#include<iostream>
#define SIZE 50
#define FILTER_RADIUS 2

__global__ void conv1d_Basic(float* N,float* P,float* F){
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    float pValue = 0;
    for(int j=-FILTER_RADIUS;j<=FILTER_RADIUS;j++){
        if(i+j>=0 && i+j<SIZE){
            pValue += N[i+j]*F[FILTER_RADIUS+j];
        }
    }
    P[i] = pValue;
}

int main(){
    float* N,*P,*F;
    N = (float*)malloc(SIZE*sizeof(float));
    P = (float*)malloc(SIZE* sizeof(float));
    F = (float*)malloc((2*FILTER_RADIUS+1)*sizeof(float));
    float *N_d,*P_d,*F_d;
    for(int i=0;i<SIZE;i++){
        N[i] = rand()%10;
    }
    for(int i=0;i<2*FILTER_RADIUS+1;i++)
        F[i] = rand()%10;
    cudaMalloc((void**)&N_d,SIZE*sizeof(float));
    cudaMalloc((void**)&P_d,SIZE*sizeof(float));
    cudaMalloc((void**)&F_d,(2*FILTER_RADIUS+1)*sizeof(float));

    cudaMemcpy(N_d,N,SIZE*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(F_d,F,(2*FILTER_RADIUS+1)*sizeof(float),cudaMemcpyHostToDevice);
    dim3 dimGrid(ceil((float)SIZE/32),1,1);
    dim3 dimBlock(32,1,1);
    conv1d_Basic<<<dimGrid,dimBlock>>>(N_d,P_d,F_d);
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
    cudaFree(F_d);
    cudaFree(P_d);
    return 0;
}