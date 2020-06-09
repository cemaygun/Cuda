#ifdef __CUDACC__
#define KERNEL_ARGS2(grid, block) <<< grid, block >>>
#define KERNEL_ARGS3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define KERNEL_ARGS4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#else
#define KERNEL_ARGS2(grid, block)
#define KERNEL_ARGS3(grid, block, sh_mem)
#define KERNEL_ARGS4(grid, block, sh_mem, stream)
#endif
float a[1024][1024], b[1024][1024], c[1024][1024];
#include<time.h>




// Now launch your kernel using the appropriate macro:

// Now launch your kernel using the appropriate macro:
//kernel KERNEL_ARGS2(dim3(nBlockCount), dim3(nThreadCount)) (param1);
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#include <stdlib.h>
#include <math.h>


//matrix multiplication on GPU
__global__ void MMul(float*m, float*d, float*p, int n) {
	int r = blockIdx.y*blockDim.y + threadIdx.y;// row 
	int c = blockIdx.x*blockDim.x + threadIdx.x;//column
	float p_sum = 0;

	for (int i = 0; i < n; i++) {
		p_sum = +m[r*n + i] * d[i*n + c];
	}
	p[r*n + c] = p_sum;
}
double get_CPU_time_usage(clock_t clock1, clock_t clock2)
{
	double diffticks = clock1 - clock2;
	double diffms = (diffticks * 1000) / CLOCKS_PER_SEC;
	return diffms;
}

int main(int argc, char* argv[]) {
	const int n = 1 << 10; //1024 square matrix size
	cudaEvent_t start, stop;
	clock_t tStart = clock();
	float sum = 0;
	float sum_p = 0;
	//Host matrix
	float* h_m;
	float* h_n;
	float* h_p;//resultant matrix on host (CPU)
			   //Device matrix
	float* d_m;
	float* d_n;
	float* d_p;// resultant matrix on device
	size_t bytes = n *n * sizeof(int);
	double cpu_time_used;
	float ms;



	//memory allocation on host mem.
	h_m = (float*)malloc(bytes);
	h_n = (float*)malloc(bytes);
	h_p = (float*)malloc(bytes);

	//matrix init.
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			h_m[i*n + j] = 3.0;
			h_n[i*n + j] = 2.0;

		}
	}
	// Measure the execution time for matrix multiplication on CPU
	clock_t begin = clock(); // clock starts 
							 /*Matrix multiplication on mem. (can be written in a function)*/
	for (int c = 0; c < n; c++) { // Fill Arrays
		for (int r = 0; r < n; r++)
		{
			for (int i = 0; i < n; i++) {
				sum = +h_m[r*n + i] * h_n[i*n + c];
			}
			h_p[r*n + c] = sum;
		}
	}
	clock_t end = clock();//clock ends

						  /* print the result */
	printf("The sum for is: \n");
	for (int i = 0; i < n; i++) {
		printf("%.1f ", h_p[i]);
	}

	/*Measure the execution time for matrix multiplication on GPU*/
	/*Allocate vectors in device memory */
	cudaMalloc(&d_m, bytes);
	cudaMalloc(&d_n, bytes);
	cudaMalloc(&d_p, bytes);

	/* Copy data from host memory to device memory */
	cudaMemcpy(d_m, h_m, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_n, h_n, bytes, cudaMemcpyHostToDevice);

	int threads_per_block = 16;

	cudaEventCreate(&start);
	cudaEventRecord(start);


	dim3 block_size(threads_per_block, threads_per_block);// creates block  which is 16x16, and in block consist of total 256 threads
	dim3 grid_size(n / block_size.x, n / block_size.y); // since our matrix size is 1024x1024, this function makes 4x4 grid.

														/* Kernel Call */
	MMul << <grid_size, block_size >> >(d_m, d_n, d_p, n);

	cudaEventCreate(&stop);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms, start, stop);
	cudaMemcpy(h_p, d_p, bytes, cudaMemcpyDeviceToHost);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	/*print the result*/
	printf("The sum is: \n");
	for (int i = 0; i < n; i++) {
		printf("%.1f ", h_p[i]);
	}



	/* Free device memory */
	cudaFree(d_m);
	cudaFree(d_n);
	cudaFree(d_p);
	/* Free host memory */
	free(h_m);
	free(h_n);
	free(h_p);
	printf("GPU: %f ms", ms);
	printf("\n CPU : %f ms", double(get_CPU_time_usage(end, begin)));
	return 0;
}  /* main */