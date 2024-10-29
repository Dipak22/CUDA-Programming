#include<iostream>
#define HEIGHT 16
#define WIDTH 16
#define RADIUS 2
__global__ void Conv2D_Basic(float* N,float* P,float* F,int filter_width){
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    if(col<WIDTH && row<HEIGHT){
        float pValue = 0.0f;
        for(int r = 0;r<filter_width;r++){
            for(int c=0;c<filter_width;c++){
                int inCol = col - RADIUS + c;
                int inRow = row - RADIUS + r;
                if(inCol>=0 && inCol<WIDTH && inRow>=0 && inRow<HEIGHT){
                    pValue +=N[inRow*WIDTH+inCol] * F[r*filter_width + c];
                }
            }
        }
        P[row*WIDTH+col] = pValue;
    }
    
}

int main(){
    float *N,*P,*F;
    int filter_width = 2*RADIUS+1;
    N = (float*)malloc(HEIGHT*WIDTH*sizeof(float));
    P = (float*)malloc(HEIGHT*WIDTH*sizeof(float));
    F = (float*)malloc(filter_width*filter_width*sizeof(float));

    for(int i=0;i<HEIGHT;i++){
        for(int j=0;j<WIDTH;j++)
            N[i*WIDTH+j] = rand()%5;
    }
    for(int i=0;i<filter_width;i++){
        for(int j=0;j<filter_width;j++){
            F[i*filter_width+j] = rand()%5;
        }
    }

    float *N_d,*F_d,*P_d;
    cudaMalloc((void**)&N_d, HEIGHT*WIDTH*sizeof(float));
    cudaMalloc((void**)&P_d, HEIGHT*WIDTH*sizeof(float));
    cudaMalloc((void**)&F_d,filter_width*filter_width*sizeof(float));

    cudaMemcpy(N_d,N,HEIGHT*WIDTH*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(F_d,F,filter_width*filter_width*sizeof(float), cudaMemcpyHostToDevice);
    dim3 dimBlock(16,16,1);
    dim3 dimGrid(ceil(WIDTH/16.0),ceil(HEIGHT/16.0),1);

    Conv2D_Basic<<<dimGrid,dimBlock>>>(N_d,P_d,F_d,filter_width);

    cudaMemcpy(P,P_d,HEIGHT*WIDTH*sizeof(float),cudaMemcpyDeviceToHost);
    std::cout<<"Input: \n";
    for(int i=0;i<HEIGHT;i++){
        for(int j=0;j<WIDTH;j++)
            std::cout<<N[i*WIDTH+j]<<" ";
        std::cout<<'\n';
    }
    std::cout<<"\n Filter:\n";
    for(int i=0;i<filter_width;i++){
        for(int j=0;j<filter_width;j++){
            std::cout<<F[i*filter_width+j]<<" ";
        }
        std::cout<<"\n";
    }
    std::cout<<"\n Output: \n";
    for(int i=0;i<HEIGHT;i++){
        for(int j=0;j<WIDTH;j++)
            std::cout<<P[i*WIDTH+j]<<" ";
        std::cout<<'\n';
    }

    free(N);
    free(P);
    free(F);

    cudaFree(N_d);
    cudaFree(P_d);
    cudaFree(F_d);
    return 0;
}