#include <stdio.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
//Code written by Alan Fleming

//CONSTANTS
#define MATRIXSIZE 2048
#define BLOCKSIZE 1024

void spmvCPU(int num_row, const float* Value, const int* col_idx, const int* row_ptr, const float* x, float* y) {
	
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
