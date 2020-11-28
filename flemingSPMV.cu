#include <stdio.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
//Code written by Alan Fleming

//CONSTANTS
#define MATRIXSIZE 2048
#define BLOCKSIZE 1024

void spmvCPU(unsigned int num_row, const float* Value, const unsigned int* col_idx, const unsigned int* row_ptr, const float* x, float* y) {
	
	//preform multiplication using CSR format
	//loop over rows
	for(int i=0; i < num_row, i++) {
		float sum = 0;
		//loop over non-zero elements
		for(int = row_ptr[i]; j < row_ptr[i + 1]; j++){
			sum += value[j] * x[col_idx[j]];
		}
		y[i] = sum;
	}
}

__global__ void spmvCuda(unsigned int num_row, const float* Value, const int* col_idx, const int* row_ptr, const float* x, float* y){

	//calculate row to work on
	int row = blockDim.x * blockIdx.x + threadIdx.x;
	//If the row is within bounds preform multiplication for that row
	if(row < num_rows) {
		float sum = 0;
		for(int = row_ptr[i]; j < row_ptr[i + 1]; j++){
			sum += value[j] * x[col_idx[j]];
		}
		y[row] = sum;
	}
}

int main( int argc, char** argv) {
	if(argc != 2) {
		printf("Please specify exactly 1 input file");
		return 1;
	}

}
