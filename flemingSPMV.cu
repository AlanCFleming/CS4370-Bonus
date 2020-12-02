#include <stdio.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <iostream>
#include <fstream>
//Code written by Alan Fleming
//CONSTANTS
#define BLOCKSIZE 1024

void spmvCPU(int num_row, const float* value, const int* col_idx, const int* row_ptr, const float* x, float* y) {
	
	//preform multiplication using CSR format
	//loop over rows
	for(int i=0; i < num_row; i++) {
		float sum = 0;
		//loop over non-zero elements
		for(int j = row_ptr[i]; j < row_ptr[i + 1]; j++){
			sum += value[j] * x[col_idx[j]];
		}
		y[i] = sum;
	}
}

__global__ void spmvCuda(int num_row, const float* value, const int* col_idx, const int* row_ptr, const float* x, float* y){

	//calculate row to work on
	int row = blockDim.x * blockIdx.x + threadIdx.x;
	//If the row is within bounds preform multiplication for that row
	if(row < num_row) {
		float sum = 0;
		for(int i = row_ptr[row]; i < row_ptr[row + 1]; i++){
			sum += value[i] * x[col_idx[i]];
		}
		y[row] = sum;
	}
}

using namespace std;

int main( int argc, char** argv) {
	if(argc != 2) {
		printf("Please specify exactly 1 input file");
		return 1;
	}

	ifstream input(argv[1]);

	if (!input.is_open()){
		printf("Data file failed to open");
		return 2;
	}
	
	//variables for counts
	int num_row, num_non_zero, num_col;

	//first line of input is num_row+1
	input >> num_row;
	num_row--;

	//Second line of input is the number of non-zero elements (the number of elements in value)
       	input >> num_non_zero;

	//Third line is number of column in matrix / rows in vector
	input >> num_col;

	
	//variables for data
	int *row_ptr = (int*)malloc(sizeof(int) * (num_row+1));
	int *col_idx = (int *)malloc(sizeof(int) * num_col);
	float *value = (float *)malloc(sizeof(float) * num_non_zero);
	float *x = (float *)malloc(sizeof(float) * num_col);

	//read in data
	for(int i = 0; i < num_row + 1; i++){
		input >> row_ptr[i];
	}
	for(int i = 0; i < num_col; i++){
		input >> col_idx[i];
	}
	for(int i = 0; i < num_non_zero; i++){
		input >> value[i];
	}
	for(int i = 0; i < num_col; i++){
		input >> x[i];
	}

	input.close();

	//declare and allocate output variables
	float *resultCPU = (float *)malloc(sizeof(float) * num_row);
	float *resultGPU = (float *)malloc(sizeof(float) * num_row);

	//Test CPU
	//Get start time
	clock_t t1 = clock();
	//Calculate reduction
		
	spmvCPU(num_row, value, col_idx, row_ptr, x, resultCPU);
				
	//Get stop time
	clock_t t2 = clock();
	//Calculate runtime
	float cpuTime= (float(t2-t1)/CLOCKS_PER_SEC*1000);

	
	
	//allocate needed memory on the gpu
	int* dev_col, dev_row;
	float* dev_value, dev_x, dev_y;
	cudaMalloc((void **)(&dev_col), sizeof(int) * num_col);
	cudaMalloc((void **)(&dev_row), sizeof(int) * (num_row + 1));
	cudaMalloc((void **)(&dev_value), sizeof(float) * num_non_zero);
	cudaMalloc((void **)(&dev_x), sizeof(float) * num_col);
	cudaMalloc((void **)(&dev_y), sizeof(float) * num_row);

	cudaMemcpy(dev_col, col_idx, sizeof(int) * num_col, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_row, row_ptr, sizeof(int) * (num_row + 1), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_value, value, sizeof(float) * num_non_zero, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_x, x, sizeof(float) * num_col, cudaMemcpyHostToDevice);

	//calculate dimensions for gpu
	dim3 dimBlock(BLOCKSIZE);
	dim3 dimGrid(ceil(double(num_row)/dimBlock.x));

	//Set up cuda events for recording runtime
	cudaEvent_t start,stop;
	float GPUTime; 
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);
											
	//calculate histogram without shared memory
	spmvCuda<<<dimGrid, dimBlock>>>(num_row, dev_value, dev_col, dev_row, dev_x, dev_y);
													
	//calculate runtime 
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&GPUTime,start,stop);

	//destroy cuda events
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	//print results
	printf("--%s--\nCPU Runtime: %f\nGpu Runtime: %f\nSpeedup: %f\n", argv[1], (double)cpuTime, (double)GPUTime, double(cpuTime / GPUTime));

	//copy resulting calculation from device
	cudaMemcpy(dev_y, resultGPU, sizeof(float) * num_row, cudaMemcpyDeviceToHost);

	//verify results
	bool valid = true;
	for(int i = 0; i < 256; i++) {	
		if(resultCPU[i] != resultGPU[i]) {
			valid = false;
			break;
		}
	}

	if(valid) {
		printf("TEST PASSED\n");
	} else {
		printf("TEST FAILED\n");
	}

	//free up memory before returning
	free(row_ptr);
	free(col_idx);
	free(value);
	free(x);
	free(resultCPU);
	free(resultGPU);
	cudaFree(dev_col);
	cudaFree(dev_row);
	cudaFree(dev_value);
	cudaFree(dev_x);
	cudaFree(dev_y);

	return 0;


}
