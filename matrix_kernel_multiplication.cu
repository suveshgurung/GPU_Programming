#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda.h>

#define KERNEL_DIM 3

__global__ void matrix_kernel_multiplication(int *A, int *K, int *O, int n, int out_dim) {
	__shared__ int s_K[KERNEL_DIM * KERNEL_DIM];

	int thread_global_id = threadIdx.y * blockDim.x + threadIdx.x;
	if (thread_global_id < KERNEL_DIM * KERNEL_DIM) {
		s_K[thread_global_id] = K[thread_global_id];
	}

	__syncthreads();

	/* Output Column (j) -> threadIdx.x
	*  Output Row (i) -> threadIdx.y
	*/
	int sum = 0;
	for (int u = 0; u < KERNEL_DIM; u++) {
		for (int v = 0; v < KERNEL_DIM; v++) {
			int row = 3 * threadIdx.y + u;
			int col = 3 * threadIdx.x + v;
			int val_A = (row < n && col < n) ? A[row * n + col] : 0;
			sum += val_A * s_K[u * KERNEL_DIM + v];
		}
	}
	O[threadIdx.y * out_dim + threadIdx.x] = sum;
}

int main(int argc, char *argv[]) {
	if (argc < 2) {
		printf("Usage: %s -n=size_of_matrix\n", argv[0]);
		return 1;
	}

	int n;
	if (strncmp("-n=", argv[1], 3) == 0) {
		n = atoi(&argv[1][3]);
	}

	/* Row major flattened 2D array */
	int *A = (int *)malloc(n * n * sizeof(int));
	int *kernel_A;
	cudaMalloc(&kernel_A, n * n * sizeof(int));

	int K[KERNEL_DIM * KERNEL_DIM];
	int *kernel_K;
	cudaMalloc(&kernel_K, KERNEL_DIM * KERNEL_DIM * sizeof(int));

	int output_dim = ceil(n / 3.0);
	int *O = (int *)malloc(output_dim * output_dim * sizeof(int));
	int *kernel_O;
	cudaMalloc(&kernel_O, output_dim * output_dim * sizeof(int));

	printf("Enter the elements of matrix A:\n");
	for (int r = 0; r < n; r++) {
		printf("Row %d:\n", (r + 1));
		for (int c = 0; c < n; c++) {
			scanf("%d", &A[r * n + c]);
		}
	}

	printf("Enter the elements of kernel matrix K:\n");
	for (int r = 0; r < KERNEL_DIM; r++) {
		printf("Row %d:\n", (r + 1));
		for (int c = 0; c < KERNEL_DIM; c++) {
			scanf("%d", &K[r * KERNEL_DIM + c]);
		}
	}

	cudaMemcpy(kernel_A, A, n * n * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(kernel_K, K, KERNEL_DIM * KERNEL_DIM * sizeof(int), cudaMemcpyHostToDevice);
	dim3 block(n, n, 1);
	matrix_kernel_multiplication<<<1, block>>>(kernel_A, kernel_K, kernel_O, n, output_dim);

	cudaMemcpy(O, kernel_O, output_dim * output_dim * sizeof(int), cudaMemcpyDeviceToHost);

	for (int r = 0; r < output_dim; r++) {
		for (int c = 0; c < output_dim; c++) {
			printf("%d\t", O[r * output_dim + c]);
		}
		printf("\n");
	}

	cudaFree(kernel_O);
	cudaFree(kernel_K);
	cudaFree(kernel_A);
	free(O);
	free(A);
	return 0;
}
