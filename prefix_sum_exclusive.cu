#include<iostream>
#define LENGTH 100
#define BLOCK_DIM 256
__global__ void prefix_sum(int* input,int* output){
    __shared__ int input_s[BLOCK_DIM];
    int index = blockIdx.x*blockDim.x+ threadIdx.x;
    if(index!=0 && index<LENGTH)
        input_s[threadIdx.x] = input[index-1];
    else
        input_s[threadIdx.x] = 0.0f;
    for(unsigned int stride = 1;stride<blockDim.x;stride *=2){
        __syncthreads();
        int temp =0;
        if(threadIdx.x>=stride)
            temp = input_s[threadIdx.x] + input_s[threadIdx.x-stride];
        __syncthreads();
        if(threadIdx.x>=stride)
            input_s[threadIdx.x] = temp;
    }
    if(index<LENGTH)
        output[index] = input_s[threadIdx.x];
}

int main(){
    int *input,*output;
    input = (int*)malloc(LENGTH*sizeof(int));
    output = (int*)malloc(LENGTH*sizeof(int));
    for(int i=0;i<LENGTH;i++)
        input[i] = rand()%10;
    int *input_d,*output_d;
    cudaMalloc((void**)&input_d,LENGTH*sizeof(int));
    cudaMalloc((void**)&output_d,LENGTH*sizeof(int));
    cudaMemcpy(input_d,input,LENGTH*sizeof(int),cudaMemcpyHostToDevice);
    dim3 dimBlock(BLOCK_DIM,1,1);
    dim3 dimGrid((LENGTH+BLOCK_DIM-1)/BLOCK_DIM,1,1);
    prefix_sum<<<dimGrid,dimBlock>>>(input_d,output_d);
    cudaMemcpy(output,output_d,LENGTH*sizeof(int),cudaMemcpyDeviceToHost);
    std::cout<<"gpu input\n";
    for(int i=0;i<LENGTH;i++)
        std::cout<<input[i]<<" ";
    std::cout<<"\ngpu output\n";
    for(int i=0;i<LENGTH;i++)
        std::cout<<output[i]<<" ";
    free(input);
    free(output);
    cudaFree(input_d);
    cudaFree(output_d);
    return 0;
}