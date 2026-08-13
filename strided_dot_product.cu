#include <stdio.h>
#include <cuda.h>

__global__ void strided_dot_product(int *A, int *B, int *R, int n, int k) {
	for (int t = 0; t < k; t++) {
		int index = (threadIdx.x + t) % n;
		R[threadIdx.x] += (A[index] * B[index]);
	}
}

int main(int argc, char *argv[]) {
	if (argc < 3) {
		printf("Usage: %s -n=size_of_vector -k=constant\n", argv[0]);
		return 1;
	}

	int n, k;
	for (int i = 1; i < argc; i++) {
		if (strncmp("-n=", argv[i], 3) == 0) {
			n = atoi(&argv[i][3]);
		}
		else if (strncmp("-k=", argv[i], 3) == 0) {
			k = atoi(&argv[i][3]);
		}
	}

	if ((k <= 0) || (k >= n)) {
		printf("Enter value of k in the range: 1 <= k <= n-1\n");
		return 0;
	}

	int *A = (int *)malloc(n * sizeof(int));
	int *B = (int *)malloc(n * sizeof(int));
	int *R = (int *)malloc(n * sizeof(int));
	int *kernel_A, *kernel_B, *kernel_R;

	cudaMalloc(&kernel_A, n * sizeof(int));
	cudaMalloc(&kernel_B, n * sizeof(int));
	cudaMalloc(&kernel_R, n * sizeof(int));

	printf("Enter elements of vector A:\n");
	for (int i = 0; i < n; i++) {
		scanf("%d", &A[i]);
	}
	printf("\nEnter elements of vector B:\n");
	for (int i = 0; i < n; i++) {
		scanf("%d", &B[i]);
	}

	for (int i = 0; i < n; i++) {
		R[i] = 0;
	}

	cudaMemcpy(kernel_A, A, n * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(kernel_B, B, n * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(kernel_R, R, n * sizeof(int), cudaMemcpyHostToDevice);

	strided_dot_product<<<1, n>>>(kernel_A, kernel_B, kernel_R, n, k);
	cudaMemcpy(R, kernel_R, n * sizeof(int), cudaMemcpyDeviceToHost);

	printf("The reuslt is:\n");
	for (int i = 0; i < n; i++) {
		printf("%d\t", R[i]);
	}
	printf("\n");
	
	cudaFree(kernel_R);
	cudaFree(kernel_B);
	cudaFree(kernel_A);
	free(R);
	free(B);
	free(A);
	return 0;
}
