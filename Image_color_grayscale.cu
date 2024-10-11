#include<iostream>
#define NUM_CHANNELS 3
#define HEIGHT 62
#define WIDHT 76

__global__ void color_to_grayscale(unsigned char *P_IN, unsigned char *P_OUT){
    int r = blockIdx.y*blockDim.y + threadIdx.y;
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    if(r<HEIGHT && c<WIDHT){
        int gray_offset = r*WIDHT + c;
        int color_offset = gray_offset*NUM_CHANNELS;
        P_OUT[gray_offset] = 0.21*P_IN[color_offset] + 0.72*P_IN[color_offset+1] + 0.07*P_IN[color_offset+2];
    }
}
int main(){
    unsigned char *h_IN, *h_OUT;
    h_IN = (unsigned char*)malloc(HEIGHT*WIDHT*NUM_CHANNELS*sizeof(char));
    h_OUT = (unsigned char*)malloc(HEIGHT*WIDHT*sizeof(char));
    for(int h =0;h<HEIGHT;h++){
        for(int w =0;w<WIDHT;w++){
            for(int c =0;c<NUM_CHANNELS;c++){
                h_IN[(h*WIDHT+w)*NUM_CHANNELS + c] = std::rand()/256;
            }
        }
    }

    unsigned char *d_IN,*d_OUT;
    cudaMalloc((void**)&d_IN,HEIGHT*WIDHT*NUM_CHANNELS*sizeof(char));
    cudaMalloc((void**)&d_OUT,HEIGHT*WIDHT*sizeof(char));

    cudaMemcpy(d_IN,h_IN,HEIGHT*WIDHT*NUM_CHANNELS*sizeof(char),cudaMemcpyHostToDevice);

    dim3 dimBlock(16,16);
    dim3 dimGrid(ceil((float)WIDHT/16.0),ceil((float)HEIGHT/16.0));
    color_to_grayscale<<<dimGrid,dimBlock>>>(d_IN,d_OUT);

    cudaMemcpy(h_OUT,d_OUT,HEIGHT*WIDHT*sizeof(char),cudaMemcpyDeviceToHost);

    for(int h =0;h<HEIGHT;h++){
        for(int w =0;w<WIDHT;w++){
            std::cout<<(int)h_OUT[h*WIDHT+w]<<" ";
        }
        std::cout<<std::endl;
    }

    cudaFree(h_IN);
    cudaFree(h_OUT);
    free(d_IN);
    free(d_OUT);

    return 0;

}