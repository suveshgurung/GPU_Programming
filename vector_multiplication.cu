#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

__global__ void vector_multiplication(int *A, int *B, int *ID_a, int *ID_b, int *result, int n) {
	int idx_a = ID_a[threadIdx.x];
	int idx_b = ID_b[threadIdx.x];
	result[threadIdx.x] = A[idx_a] * B[idx_b];
}

void generate_random_index(int *vector, int n) {
	for (int i = 0; i < n; i++) {
		vector[i] = i;
	}

	for (int i = n - 1; i > 0; i--) {
		int j = rand() % (i + 1);

		int temp = vector[i];
		vector[i] = vector[j];
		vector[j] = temp;
	}
}

int main(int argc, char *argv[]) {
	if (argc < 2) {
		printf("Usage: %s -n=size_of_vector\n", argv[0]);
		return 1;
	}

	if (strncmp("-n=", argv[1], 3) != 0) {
		printf("Usage: %s -n=size_of_vector\n", argv[0]);
		return 1;
	}

	srand(time(NULL));
	int n = atoi(&argv[1][3]);
	
	int *A = (int *)malloc(n * sizeof(int));
	int *B = (int *)malloc(n * sizeof(int));
	int *ID_a = (int *)malloc(n * sizeof(int));
	int *ID_b = (int *)malloc(n * sizeof(int));
	int *result = (int *)malloc(n * sizeof(int));

	int *kernel_A;
	cudaMalloc(&kernel_A, n * sizeof(int));
	int *kernel_B;
	cudaMalloc(&kernel_B, n * sizeof(int));
	int *kernel_ID_a;
	cudaMalloc(&kernel_ID_a, n * sizeof(int));
	generate_random_index(ID_a, n);
	cudaMemcpy(kernel_ID_a, ID_a, n * sizeof(int), cudaMemcpyHostToDevice);
	int *kernel_ID_b;
	cudaMalloc(&kernel_ID_b, n * sizeof(int));
	generate_random_index(ID_b, n);
	cudaMemcpy(kernel_ID_b, ID_b, n * sizeof(int), cudaMemcpyHostToDevice);
	int *kernel_result;
	cudaMalloc(&kernel_result, n * sizeof(int));
	
	printf("Enter elements of matrix A:\n");
	for (int i = 0; i < n; i++) {
		scanf("%d", &A[i]);
	}
	printf("\nEnter elements of matrix B:\n");
	for (int i = 0; i < n; i++) {
		scanf("%d", &B[i]);
	}

	cudaMemcpy(kernel_A, A, n * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(kernel_B, B, n * sizeof(int), cudaMemcpyHostToDevice);

	/* Launch the kernel */
	vector_multiplication<<<1, n>>>(kernel_A, kernel_B, kernel_ID_a, kernel_ID_b, kernel_result, n);
	cudaMemcpy(result, kernel_result, n * sizeof(int), cudaMemcpyDeviceToHost);

	printf("The values being multiplied are:\n");
	for (int i = 0; i < n; i++) {
		printf("A[%d] * B[%d]\n", ID_a[i], ID_b[i]);
	}

	printf("The multiplied vector is:\n");
	for (int i = 0; i < n; i++) {
		printf("%d\t", result[i]);
	}

	free(result);
	free(ID_b);
	free(ID_a);
	free(B);
	free(A);
	cudaFree(kernel_A);
	cudaFree(kernel_B);
	cudaFree(kernel_ID_a);
	cudaFree(kernel_ID_b);
	cudaFree(kernel_result);
	printf("\n");
	return 0;
}
