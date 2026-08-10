#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

__global__ void vector_addition(int *A, int *B) {
	A[threadIdx.x] += B[threadIdx.x];
}

int main(int argc, char *argv[]) {
	if (argc < 2) {
		printf("Usage: %s -n=number_of_elements\n", argv[0]);
		return 1;
	}

	/* n -> size of the array */
	int n;
	if (strncmp(argv[1], "-n=", 3) == 0) {
		n = atoi(&argv[1][3]);
	}

	int *kernel_A;
	int *kernel_B;
	cudaMalloc(&kernel_A, n * sizeof(int));
	cudaMalloc(&kernel_B, n * sizeof(int));

	int *host_A = (int *)malloc(n * sizeof(int));
	int *host_B = (int *)malloc(n * sizeof(int));
	int *result = (int *)malloc(n * sizeof(int));

	printf("Enter elements of vector A:\n");
	for (int i = 0; i < n; i++) {
		scanf("%d", &host_A[i]);
	}
	printf("Enter elements of vector B:\n");
	for (int i = 0; i < n; i++) {
		scanf("%d", &host_B[i]);
	}

	cudaMemcpy(kernel_A, host_A, n * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(kernel_B, host_B, n * sizeof(int), cudaMemcpyHostToDevice);

	vector_addition<<<1, n>>>(kernel_A, kernel_B);
	cudaMemcpy(result, kernel_A, n * sizeof(int), cudaMemcpyDeviceToHost);

	printf("The elements after addition are:\n");
	for (int i = 0; i < n; i++) {
		printf("%d\t", result[i]);
	}
	printf("\n");

	cudaFree(kernel_A);
	cudaFree(kernel_B);
	free(host_A);
	free(host_B);
	free(result);
	return 0;
}
