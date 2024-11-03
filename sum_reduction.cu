#include<iostream>
#define LENGTH 2048

__global__ void sum_kernel(int *input,int *output){
    for(unsigned int stride = 1;stride<LENGTH;stride *=2){
        int index = 2*stride*threadIdx.x;
        if( index<LENGTH){
            input[index] +=input[index+stride];
        }
        __syncthreads();
    }
    if(threadIdx.x==0)
        *output = input[0];
}

int main(){
    int *input,*sum;
    input = (int*)malloc(LENGTH*sizeof(int));
    sum = (int*)malloc(sizeof(int));
    for(int i=0;i<LENGTH;i++)
        input[i] = rand()%10;
    int *input_d,*output;
    cudaMalloc((void**)&input_d,LENGTH*sizeof(int));
    cudaMalloc((void**)&output,sizeof(int));
    cudaMemcpy(input_d,input,LENGTH*sizeof(int),cudaMemcpyHostToDevice);
    dim3 dimBlock(LENGTH/2,1,1);
    dim3 dimGrid(1,1,1);
    sum_kernel<<<dimGrid,dimBlock>>>(input_d,output);
    cudaMemcpy(sum,output,sizeof(int),cudaMemcpyDeviceToHost);
    std::cout<<"Input : \n";
    for(int i=0;i<LENGTH;i++)
        std::cout<<input[i]<<" ";

    std::cout<<"\nsum : "<<*sum;
    int cpu_sum = 0;
    for(int i=0;i<LENGTH;i++)
        cpu_sum +=input[i];
    std::cout<<"\ncpu sum : "<<cpu_sum;
    free(input);
    cudaFree(input_d);
    cudaFree(output);
    return 0;
}