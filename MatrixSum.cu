#include<iostream>
#define ROWS 20
#define COLS 20
__global__ void mat_sum_2DBlock(float *A,float *B,float* C){
    int r = blockIdx.y*blockDim.y + threadIdx.y;
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    if(r<ROWS && c<COLS){
        C[r*COLS+c] = A[r*COLS+c] + B[r*COLS+c];
    }
}

__global__ void mat_sum_1DBlock(float *A,float *B,float* C){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<ROWS*COLS){
        C[i] = A[i]+B[i];
    }
}

int main(){
    float* h_a,*h_b,*h_c;
    float *d_a,*d_b,*d_c;
    h_a = (float*)malloc(ROWS * COLS * sizeof(float));
    h_b = (float*)malloc(ROWS * COLS * sizeof(float));
    h_c = (float*)malloc(ROWS * COLS * sizeof(float));

    for(int i =0;i<ROWS;i++){
        for(int j=0;j<COLS;j++){
            h_a[i*COLS+j] = 5.0f;
            h_b[i*COLS+j] = 10.0f;
            h_c[i*COLS+j] = 0.0f;
        }
    }

    cudaMalloc((void**)&d_a,ROWS*COLS*sizeof(float));
    cudaMalloc((void**)&d_b,ROWS*COLS*sizeof(float));
    cudaMalloc((void**)&d_c,ROWS*COLS*sizeof(float));

    cudaMemcpy(d_a,h_a,ROWS*COLS*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,ROWS*COLS*sizeof(float),cudaMemcpyHostToDevice);
    //dim3 dimBlock(32,32);
    //dim3 dimGrid(ceil((float)ROWS/32),ceil((float)COLS/32));
    //mat_sum_2DBlock<<<dimGrid,dimBlock>>>(d_a,d_b,d_c);
    dim3 dimBlock(32);
    dim3 dimGrid(ceil((float)(ROWS*COLS)/32));
    mat_sum_1DBlock<<<dimGrid,dimBlock>>>(d_a,d_b,d_c);
    cudaMemcpy(h_c,d_c,ROWS*COLS*sizeof(float),cudaMemcpyDeviceToHost);
    std::cout<<"\n Matrix C \n";
    for(int i =0;i<ROWS;i++){
        for(int j=0;j<COLS;j++){
            std::cout<<h_c[i*COLS+j]<<" ";
            //std::cout<<(i*COLS+j)<<" ";
        }
        std::cout<<std::endl;
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}