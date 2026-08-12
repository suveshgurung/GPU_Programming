#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>

__global__ void matrix_transpose(int **matrix, int **transpose_matrix, int m, int n) {
	int row = threadIdx.x;
	int col = threadIdx.y;
	transpose_matrix[col][row] = matrix[row][col];
}

int main(int argc, char *argv[]) {
	if (argc < 3) {
		printf("Usage: %s -m=no_of_rows -n=no_of_cols\n", argv[0]);
		return 1;
	}

	int m, n;
	for (int i = 1; i < argc; i++) {
		if (strncmp(argv[i], "-m=", 3) == 0) {
			m = atoi(&argv[i][3]);
		}
		else if (strncmp(argv[i], "-n=", 3) == 0) {
			n = atoi(&argv[i][3]);
		}
	}

	int **matrix = (int **)malloc(m * sizeof(int *));
	for (int i = 0; i < m; i++) {
		matrix[i] = (int *)malloc(n * sizeof(int));
	}

	int **kernel_matrix;
	cudaMalloc(&kernel_matrix, m * sizeof(int *));
	int **host_kernel_matrix = (int **)malloc(m * sizeof(int *));
	for (int i = 0; i < m; i++) {
		cudaMalloc(&host_kernel_matrix[i], n * sizeof(int));
	}
	/* Copy the pointers to the columns to GPU */
	cudaMemcpy(kernel_matrix, host_kernel_matrix, m * sizeof(int *), cudaMemcpyHostToDevice);

	int **kernel_transpose_matrix;
	cudaMalloc(&kernel_transpose_matrix, n * sizeof(int *));
	int **host_kernel_transpose_matrix = (int **)malloc(n * sizeof(int *));
	for (int i = 0; i < n; i++) {
		cudaMalloc(&host_kernel_transpose_matrix[i], m * sizeof(int));
	}
	cudaMemcpy(kernel_transpose_matrix, host_kernel_transpose_matrix, n * sizeof(int *), cudaMemcpyHostToDevice);
	
	for (int r = 0; r < m; r++) {
		printf("Enter elements of the row %d:\n", r);
		for (int c = 0; c < n; c++) {
			scanf("%d", &matrix[r][c]);
		}
		printf("\n");
	}
	printf("The matrix you entered is:\n");
	for (int r = 0; r < m; r++) {
		for (int c = 0; c < n; c++) {
			printf("%d\t", matrix[r][c]);
		}
		printf("\n");
	}

	for (int i = 0; i < m; i++) {
		cudaMemcpy(host_kernel_matrix[i], matrix[i], n * sizeof(int), cudaMemcpyHostToDevice);
	}

	int **transpose_matrix = (int **)malloc(n * sizeof(int *));
	for (int i = 0; i < n; i++) {
		transpose_matrix[i] = (int *)malloc(m * sizeof(int));
	}
	
	dim3 block(m, n, 1);
	matrix_transpose<<<1, block>>>(kernel_matrix, kernel_transpose_matrix, m, n);
	for (int i = 0; i < n; i++) {
		cudaMemcpy(transpose_matrix[i], host_kernel_transpose_matrix[i], m * sizeof(int), cudaMemcpyDeviceToHost);
	}

	printf("The transpose matrix is:\n");
	for (int r = 0; r < n; r++) {
		for (int c = 0; c < m; c++) {
			printf("%d\t", transpose_matrix[r][c]);
		}
		printf("\n");
	}
	/* Free matrix allocated for device */
	for (int i = 0; i < n; i++) {
		cudaFree(host_kernel_transpose_matrix[i]);
	}
	cudaFree(kernel_transpose_matrix);
	free(host_kernel_transpose_matrix);
	for (int i = 0; i < m; i++) {
		cudaFree(host_kernel_matrix[i]);
	}
	cudaFree(kernel_matrix);
	free(host_kernel_matrix);
	/* Free the allocated matrix */
	for (int i = 0; i < n; i++) {
		free(transpose_matrix[i]);
	}
	free(transpose_matrix);
	for (int i = 0; i < m; i++) {
		free(matrix[i]);
	}
	free(matrix);

	return 0;
}
