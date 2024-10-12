#include<iostream>
#define WIDTH 4

__global__ void matrix_mul(float *A,float *B,float *C){
    int r = blockIdx.y*blockDim.y + threadIdx.y;
    int c= blockIdx.x*blockDim.x  +threadIdx.x;
    if(r<WIDTH && c<WIDTH){
        float sum =0;
        for(int i=0;i<WIDTH;i++){
            sum +=A[r*WIDTH+i] * B[i*WIDTH + c];
        }
        C[r*WIDTH+c] = sum;
    }
}

__global__ void mat_mul_RowWise(float *A,float *B,float *C){
    int r = blockIdx.x*blockDim.x+ threadIdx.x;
    for(int c =0;c<WIDTH;c++){
        int sum =0;
        for(int k =0;k<WIDTH;k++){
            sum +=A[r*WIDTH+k] * B[k*WIDTH + c];
        }
        C[r*WIDTH+c] = sum;
    }
}
__global__ void mat_mul_colWise(float *A,float *B,float *C){
    int c = blockIdx.x* blockDim.x + threadIdx.x;
    for(int r =0;r<WIDTH;r++){
        int sum =0;
        for(int k =0;k<WIDTH;k++){
            sum += A[r*WIDTH+k] * B[k*WIDTH + c];
        }
        C[r*WIDTH + c] = sum;
    }
}

int main(){
    float *A,*B,*C;
    A = (float*)malloc(WIDTH*WIDTH*sizeof(float));
    B = (float*)malloc(WIDTH*WIDTH*sizeof(float));
    C = (float*)malloc(WIDTH*WIDTH*sizeof(float));
    for(int i=0;i<WIDTH;i++){
        for(int j =0;j<WIDTH;j++){
            A[i*WIDTH+j] = std::rand()%5;
            B[i*WIDTH+j] = std::rand()%5;
        }
    }

    float *d_A,*d_B,*d_C;
    cudaMalloc((void**)&d_A,WIDTH*WIDTH*sizeof(float));
    cudaMalloc((void**)&d_B,WIDTH*WIDTH*sizeof(float));
    cudaMalloc((void**)&d_C,WIDTH*WIDTH*sizeof(float));

    cudaMemcpy(d_A,A,WIDTH*WIDTH*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,B,WIDTH*WIDTH*sizeof(float),cudaMemcpyHostToDevice);
    //dim3 dimBlock(16,16);
    //dim3 dimGrid(ceil(WIDTH/16.0),ceil(WIDTH/16.0));
    //matrix_mul<<<dimGrid,dimBlock>>>(d_A,d_B,d_C);
    dim3 dimBlock(16);
    dim3 dimGrid(ceil(WIDTH/16.0));
    mat_mul_colWise<<<dimGrid,dimBlock>>>(d_A,d_B,d_C);

    cudaMemcpy(C,d_C,WIDTH*WIDTH*sizeof(float),cudaMemcpyDeviceToHost);

    std::cout<<"Matrix A\n";

    for(int i=0;i<WIDTH;i++){
        for(int j =0;j<WIDTH;j++){
            std::cout<<A[i*WIDTH+j]<<" ";
        }
        std::cout<<"\n";
    }

    std::cout<<"Matrix B\n";

    for(int i=0;i<WIDTH;i++){
        for(int j =0;j<WIDTH;j++){
            std::cout<<B[i*WIDTH+j]<<" ";
        }
        std::cout<<"\n";
    }

    std::cout<<"Matrix C\n";

    for(int i=0;i<WIDTH;i++){
        for(int j =0;j<WIDTH;j++){
            std::cout<<C[i*WIDTH+j]<<" ";
        }
        std::cout<<"\n";
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C);

    return 0;
}