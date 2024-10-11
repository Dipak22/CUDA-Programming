#include<iostream>
#define ROWS 50
#define COLS 50

__global__ void mat_vec_mul_V1(float *A,float *v,float *C){
    int r = blockIdx.x*blockDim.x + threadIdx.x;
    if(r<ROWS){
        C[r] =0.0f;
        for(int i=0;i<COLS;i++){
            C[r] += A[r*COLS+i] * v[i];
        }
    }
}

int main(){
    float* h_A,*h_v,*h_C;
    h_A = (float*)malloc(ROWS*COLS*sizeof(float));
    h_v = (float*)malloc(COLS*sizeof(float));
    h_C = (float*)malloc(ROWS*sizeof(float));
    for(int i=0;i<ROWS;i++){
        for(int j = 0;j<COLS;j++)
            h_A[i*COLS+j] = 2.0f;
    }
    for(int i=0;i<COLS;i++)
        h_v[i] = 10.0f;
    for(int i=0;i<ROWS;i++)
        h_C[i] = 0.0f;
    float *d_A,*d_v, *d_C;
    cudaMalloc((void**)&d_A,ROWS*COLS*sizeof(float));
    cudaMalloc((void**)&d_v,COLS*sizeof(float));
    cudaMalloc((void**)&d_C,ROWS*sizeof(float));

    cudaMemcpy(d_A,h_A,ROWS*COLS*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_v,h_v,COLS*sizeof(float),cudaMemcpyHostToDevice);
    dim3 dimBlock(32);
    dim3 dimGrid(ceil((float)ROWS/32));
    mat_vec_mul_V1<<<dimGrid,dimBlock>>>(d_A,d_v,d_C);
    cudaMemcpy(h_C,d_C,ROWS*sizeof(float),cudaMemcpyDeviceToHost);
    for(int i=0;i<ROWS;i++)
        std::cout<<h_C[i]<<"\n";
    cudaFree(d_A);
    cudaFree(d_v);
    cudaFree(d_C);

    free(h_A);
    free(h_v);
    free(h_C);

    return 0;
}