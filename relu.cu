#include <stdio.h>
 
const int N = 16; 
const int blocksize = 1; 
 
__global__ 
void relu(int* d_in) 
{
	// map function: f(x) = x if x >= 0, 0 otherwise
	int val = d_in[threadIdx.x];
	d_out[threadIdx.x] = val < 0 ? 0 : val;
}
 
int main()
{
 
	int *h_in;
	const int size = N*sizeof(int);
 
	malloc((void**)&h_in, size); 
	malloc((void**)&h_out, size);  

	for (int ii = 0; ii < N; ii++) {
	    if (ii % 2)
		h_in[ii] = ii;
	    else
		h_in[ii] = -ii;
	}
	cudaMalloc((void**)&d_in, size); 
	cudaMalloc((void**)&d_out, size); 
	cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice); 
	
	dim3 dimBlock(blocksize, N);
	dim3 dimGrid(1, 1);
	hello<<<dimGrid, dimBlock>>>(d_in);
	cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost); 
	for (int ii = 0; ii < N; ii++)
	    printf("\nin[%d]=%d\tout[%d]=%d", ii, h_in[ii], ii, h_out[ii]);
	free(h_in);
	free(h_out);
	cudaFree(d_in);
	cudaFree(d_out);
	
	return EXIT_SUCCESS;
}
