#include<iostream>
#include<string>

__global__ void calc_hist(char *data,int length,unsigned int *hist){
    int i = blockIdx.x*blockDim.x+ threadIdx.x;
    if(i<length){
        int ch = data[i] - 'a';
        if(ch>=0 && ch<26){
            atomicAdd(&(hist[ch/4]),1);
        }
    }
}

int main(){
    std::string data_h = "programming massively parallel processors";
    int length = data_h.length();
    int hist_length = (int)26/4 + 1;
    unsigned int *hist_h;
    hist_h= (unsigned int*)malloc(hist_length*sizeof(int));
    memset(hist_h, 0, hist_length * sizeof(int));
    char *data_d;
    unsigned int *hist_d;
    cudaMalloc((void**)&data_d,length*sizeof(char));
    cudaMalloc((void**)&hist_d,hist_length*sizeof(int));
    cudaMemcpy(data_d,data_h.c_str(),length*sizeof(char),cudaMemcpyHostToDevice);
    cudaMemset(hist_d, 0, hist_length * sizeof(int));
    dim3 dimBlock(256,1,1);
    dim3 dimGrid(ceil(length/256.0),1,1);
    calc_hist<<<dimGrid,dimBlock>>>(data_d,length,hist_d);
    cudaMemcpy(hist_h,hist_d,hist_length*sizeof(int),cudaMemcpyDeviceToHost);
    std::cout<<"histogram\n";
    //char c_start = 'a';
    for(int i=0;i<hist_length;i++)
        std::cout<<hist_h[i]<<" ";

    free(hist_h);
    cudaFree(data_d);
    cudaFree(hist_d);
    return 0;
}