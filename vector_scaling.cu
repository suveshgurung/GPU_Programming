#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cuda.h>

__global__ void vector_scalar(int *A, int scalar) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	A[index] *= scalar;
}

int main(int argc, char *argv[]) {
	if (argc < 3) {
		printf("Usage: %s -n=number_of_elements -s=scalar\n", argv[0]);
		return 1;
	}

	/* n -> size of the array
	*  s -> scalar
	*/
	int n, s;
	for (int i = 1; i < argc; i++) {
		if (strncmp(argv[i], "-n=", 3) == 0) {
			n = atoi(&argv[i][3]);
		}
		else if (strncmp(argv[i], "-s=", 3) == 0) {
			s = atoi(&argv[i][3]);
		}
	}

	int *kernel_A;
	cudaMalloc(&kernel_A, n * sizeof(int));
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);	

	int *A = (int *)malloc(n * sizeof(int));
	printf("Enter elements of the array:\n");
	for (int i = 0; i < n; i++) {
		scanf("%d", &A[i]);
	}
	printf("Scaling the vector...\n");

	cudaMemcpy(kernel_A, A, n * sizeof(int), cudaMemcpyHostToDevice);
	
	cudaEventRecord(start);
	vector_scalar<<<1, n>>>(kernel_A, s);				/* Single thread block containing all of the threads. */
	// vector_scalar<<<2, n/2>>>(kernel_A, s);			/* Two thread blocks containing half of the threads each. */
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	cudaMemcpy(A, kernel_A, n * sizeof(int), cudaMemcpyDeviceToHost);
	printf("The scaled vector is:\n");
	for (int i = 0; i < n; i++) {
		printf("%d\t", A[i]);
	}
	printf("\n");

	float run_time = 0;
	cudaEventElapsedTime(&run_time, start, stop);
	printf("Kernel Time: %f ms\n", run_time);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFree(kernel_A);
	free(A);
	return 0;
}
