#include <stdio.h>
 
const int N = 16; 
 
__global__ 
void relu(int* d_in, int *d_out) 
{
	// map function: f(x) = x if x >= 0, 0 otherwise
	int val = d_in[threadIdx.x];
	d_out[threadIdx.x] = val < 0 ? 0 : val;
}
 
int main()
{
 
	int *h_in, *h_out, *d_in, *d_out;
	const int size = N*sizeof(int);
 
	h_in = (int*)malloc(size); 
	h_out = (int*)malloc(size); 

	for (int ii = 0; ii < N; ii++) {
	    if (ii % 2)
		h_in[ii] = ii;
	    else
		h_in[ii] = -ii;
	}
	cudaMalloc((void**)&d_in, size); 
	cudaMalloc((void**)&d_out, size); 
	cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice); 
	
	dim3 dimBlock(N, 1);
	dim3 dimGrid(1, 1);
	relu<<<dimGrid, dimBlock>>>(d_in, d_out);
	cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost); 
	for (int ii = 0; ii < N; ii++)
	    printf("\nin[%d]=%d\tout[%d]=%d", ii, h_in[ii], ii, h_out[ii]);
	free(h_in);
	free(h_out);
	cudaFree(d_in);
	cudaFree(d_out);
	
	return EXIT_SUCCESS;
}
