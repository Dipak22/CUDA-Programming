#include<iostream>
#define HEIGHT 10
#define WIDHT 10
#define BLUR_SIZE 1 
__global__ void blur_image(unsigned char *P_IN,unsigned char *P_OUT){
    int r = blockIdx.y*blockDim.y+threadIdx.y;
    int c = blockIdx.x*blockDim.x+ threadIdx.x;
    if(r<HEIGHT && c<WIDHT){
        int sum =0;
        int count = 0;
        for(int i = -BLUR_SIZE;i<=BLUR_SIZE;i++){
            for(int j = -BLUR_SIZE;j<=BLUR_SIZE;j++){
                int row = r+i;
                int col = c+j;
                if(row>=0 && row<HEIGHT && col>=0 && col<WIDHT){
                    sum += P_IN[row*WIDHT+col];
                    count++;
                }
            }

        }
        P_OUT[r*WIDHT+c] = (unsigned char)(sum/count);
    }
}
int main(){
    unsigned char *h_IN, *h_OUT;
    h_IN = (unsigned char*)malloc(HEIGHT*WIDHT*sizeof(char));
    h_OUT = (unsigned char*)malloc(HEIGHT*WIDHT*sizeof(char));

    for(int i=0;i<HEIGHT;i++){
        for(int j=0;j<WIDHT;j++)
            h_IN[i*WIDHT+j] = std::rand()/256;
    }

    unsigned char *d_IN, *d_OUT;
    cudaMalloc((void**)&d_IN,HEIGHT*WIDHT*sizeof(char));
    cudaMalloc((void**)&d_OUT,HEIGHT*WIDHT*sizeof(char));

    cudaMemcpy(d_IN,h_IN,HEIGHT*WIDHT*sizeof(char),cudaMemcpyHostToDevice);
    dim3 dimBlock(16,16);
    dim3 dimGrid(ceil(WIDHT/16.0),ceil(HEIGHT/16.0));
    blur_image<<<dimGrid,dimBlock>>>(d_IN,d_OUT);

    cudaMemcpy(h_OUT,d_OUT,HEIGHT*WIDHT*sizeof(char),cudaMemcpyDeviceToHost);
    for(int i=0;i<HEIGHT;i++){
        for(int j=0;j<WIDHT;j++)
            std::cout<<(int)h_IN[i*WIDHT+j]<<" ";
        std::cout<<"\n";
    }
    std::cout<<"\n\n";
    for(int i=0;i<HEIGHT;i++){
        for(int j=0;j<WIDHT;j++)
            std::cout<<(int)h_OUT[i*WIDHT+j]<<" ";
        std::cout<<"\n";
    }

    cudaFree(d_IN);
    cudaFree(d_OUT);

    free(h_IN);
    free(h_OUT);

    return 0;
}