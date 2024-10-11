#include<iostream>

#define SIZE 1000

__global__ void vecSum(int* a,int*b,int* c){
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if(i<SIZE){
		c[i] = a[i]+b[i];
	}
}

int main(){
	int h_a[SIZE],h_b[SIZE],h_c[SIZE];
	int* d_a,*d_b,*d_c;
	for(int i =0;i<SIZE;i++){
		h_a[i] = i*i;
		h_b[i] = i;
	}
	cudaMalloc((void**)&d_a,SIZE*sizeof(int));
	cudaMalloc((void**) &d_b,SIZE*sizeof(int));
	cudaMalloc((void**) &d_c,SIZE*sizeof(int));
	cudaMemcpy(d_a,h_a,SIZE*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(d_b,h_b,SIZE*sizeof(int), cudaMemcpyHostToDevice);
	vecSum<<<ceil((float)SIZE/256),256>>>(d_a,d_b,d_c);
	cudaMemcpy(h_c,d_c,SIZE*sizeof(int),cudaMemcpyDeviceToHost);
	for(int i =0;i<SIZE;i++){
		std::cout<<h_a[i]<<" + "<<h_b[i]<<" = "<<h_c[i]<<"\n";
	}
	std::cout<<'\n';
	
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	
	return 0;
}