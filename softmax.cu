#include <stdio.h>
#include <math.h>
 
const double N = 16; 
 
__global__ 
void exp(double* d_in, double *d_exp) 
{
	unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
	// map function: exp(xi)
	d_exp[idx] = exp(d_in[idx]);
}

__global__ 
void sum(double *d_exp, double *d_sum) 
{
	// reduction function: sum(exp(x))
	extern __shared__ double sdata[];
	// each thread loads one element from global to shared mem
	unsigned tid = threadIdx.x;
	unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
	sdata[tid] = d_exp[idx];
	__syncthreads();
	// do reduction in shared mem
	for (unsigned s=1; s < blockDim.x; s *= 2) {
		if (tid % (2*s) == 0) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	// write result to global mem
	if (tid == 0) *d_sum = sdata[0];
}

__global__ 
void softmax(double* d_in, double *d_sum, double *d_out) 
{
	unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
	// map function: softmax(xi) = exp(xi)/sum
	d_out[idx] = exp(d_in[idx])/(*d_sum);
}

int main()
{
 
	double *h_in;
        double *d_in, *d_exp, *d_sum, *d_out, *h_out;
	const int size = N*sizeof(double);
 
	h_in = (double*)malloc(size); 
	h_out = (double*)malloc(size); 

	for (int ii = 0; ii < N; ii++) {
		h_in[ii] = ii/N;
		printf("h_in[%d] = %f", ii, h_in[ii]);
	}
	cudaMalloc((void**)&d_in, size); 
	cudaMalloc((void**)&d_exp, size); 
	cudaMalloc((void**)&d_sum, sizeof(double)); 
	cudaMalloc((void**)&d_out, size); 
	cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice); 

	// softmax: f(xi) = exp(xi)/sum(exp(x))
	dim3 dimBlock(N, 1);
	dim3 dimGrid(N, 1);
	exp<<<dimGrid, dimBlock>>>(d_in, d_exp);
	cudaMemcpy(h_out, d_exp, size, cudaMemcpyDeviceToHost); 
	for (int ii = 0; ii < N; ii++)
		printf("exp[%d]=%f", ii, h_out[ii]);
	sum<<<dimGrid, dimBlock>>>(d_exp, d_sum);
	softmax<<<dimGrid, dimBlock>>>(d_in, d_sum, d_out);
	cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost); 
	for (int ii = 0; ii < N; ii++)
		printf("softmax[%d]=%f", ii, h_out[ii]);
	free(h_in);
	free(h_out);
	cudaFree(d_in);
	cudaFree(d_exp);
	cudaFree(d_sum);
	cudaFree(d_out);
	
	return EXIT_SUCCESS;
}
