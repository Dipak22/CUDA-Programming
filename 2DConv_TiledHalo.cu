#include<iostream>
#define WIDTH 20
#define RADIUS 2
#define MASK_WIDTH (2*RADIUS+1)
#define OUTPUT_TILE_WIDTH 16
#define INPUT_TILE_WIDTH (OUTPUT_TILE_WIDTH + MASK_WIDTH-1)
__constant__ float M[MASK_WIDTH*MASK_WIDTH];

__global__ void conv2d_tiledHalo(float *N, float *P){
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y*OUTPUT_TILE_WIDTH + ty - RADIUS;
    int col = blockIdx.x * OUTPUT_TILE_WIDTH + tx - RADIUS;
    __shared__ float N_s[INPUT_TILE_WIDTH][INPUT_TILE_WIDTH];
    if(row>=0 && row<WIDTH && col>=0 && col<WIDTH)
        N_s[ty][tx] = N[row*WIDTH+col];
    else
        N_s[ty][tx] = 0.0f;

    __syncthreads();

    int tileRow = ty - RADIUS;
    int tileCol = tx - RADIUS;
    
    if(row>=0 && row<WIDTH && col>=0 && col<WIDTH){
        if(tileRow>=0 && tileRow<OUTPUT_TILE_WIDTH && tileCol>=0 && tileCol<OUTPUT_TILE_WIDTH){
            float pValue = 0.0f;
            for(int r =0;r<MASK_WIDTH;r++){
                for(int c=0;c<MASK_WIDTH;c++){
                    pValue +=M[r*MASK_WIDTH+c] * N_s[tileRow+r][tileCol+c];
                }
            }
            P[row*WIDTH+col] = pValue;
        }
    }
}

int main(){
    float *N,*P,*M_h;
    N = (float*)malloc(WIDTH*WIDTH*sizeof(float));
    P = (float*)malloc(WIDTH*WIDTH*sizeof(float));
    M_h =(float*)malloc(MASK_WIDTH*MASK_WIDTH*sizeof(float));
    for(int i=0;i<WIDTH;i++){
        for(int j=0;j<WIDTH;j++){
            N[i*WIDTH+j] = rand()%10;
        }
    }
    for(int i=0;i<MASK_WIDTH;i++){
        for(int j=0;j<MASK_WIDTH;j++){
            M_h[i*MASK_WIDTH+j] = rand()%10;
        }
    }
    cudaMemcpyToSymbol(M,M_h,MASK_WIDTH*MASK_WIDTH*sizeof(float));

    float *N_d,*P_d;
    cudaMalloc((void**)&N_d,WIDTH*WIDTH*sizeof(float));
    cudaMalloc((void**)&P_d,WIDTH*WIDTH*sizeof(float));
    cudaMemcpy(N_d,N,WIDTH*WIDTH*sizeof(float),cudaMemcpyHostToDevice);
    dim3 dimBlock(INPUT_TILE_WIDTH,INPUT_TILE_WIDTH,1);
    dim3 dimGrid((WIDTH+OUTPUT_TILE_WIDTH-1)/OUTPUT_TILE_WIDTH,(WIDTH+OUTPUT_TILE_WIDTH-1)/OUTPUT_TILE_WIDTH,1);
    conv2d_tiledHalo<<<dimGrid,dimBlock>>>(N_d,P_d);
    cudaMemcpy(P,P_d,WIDTH*WIDTH*sizeof(float),cudaMemcpyDeviceToHost);
    std::cout<<"Input\n";
    for(int i=0;i<WIDTH;i++){
        for(int j=0;j<WIDTH;j++){
            std::cout<<N[i*WIDTH+j]<<" ";
        }
        std::cout<<"\n";
    }
    std::cout<<"\nMask:\n";
    for(int i=0;i<MASK_WIDTH;i++){
        for(int j=0;j<MASK_WIDTH;j++){
            std::cout<<M_h[i*MASK_WIDTH+j]<<" ";
        }
        std::cout<<"\n";
    }
    std::cout<<"\nOutput:\n";
    for(int i=0;i<WIDTH;i++){
        for(int j=0;j<WIDTH;j++){
            std::cout<<P[i*WIDTH+j]<<" ";
        }
        std::cout<<"\n";
    }

    free(N);
    free(P);
    free(M_h);

    return 0;
}