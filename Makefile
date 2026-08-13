CC=nvcc
OUT_DIR=outputs

test: test.cu
	mkdir -p $(OUT_DIR)
	$(CC) -o $(OUT_DIR)/test test.cu

vector_scaling: vector_scaling.cu
	mkdir -p $(OUT_DIR)
	$(CC) -o $(OUT_DIR)/vector_scaling vector_scaling.cu

vector_addition: element_wise_vector_addition.cu
	mkdir -p $(OUT_DIR)
	$(CC) -o $(OUT_DIR)/vector_addition element_wise_vector_addition.cu

matrix_transpose: matrix_transpose.cu
	mkdir -p $(OUT_DIR)
	$(CC) -o $(OUT_DIR)/matrix_transpose matrix_transpose.cu

vector_multiplication: vector_multiplication.cu
	mkdir -p $(OUT_DIR)
	$(CC) -o $(OUT_DIR)/vector_multiplication vector_multiplication.cu

strided_dot_product: strided_dot_product.cu
	mkdir -p $(OUT_DIR)
	$(CC) -o $(OUT_DIR)/strided_dot_product strided_dot_product.cu

.PHONY: clean
clean:
	rm -rf $(OUT_DIR)
