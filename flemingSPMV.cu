#include <stdio.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <fstream.h>
//Code written by Alan Fleming

//CONSTANTS
#define BLOCKSIZE 1024

void spmvCPU(unsigned int num_row, const float* value, const unsigned int* col_idx, const unsigned int* row_ptr, const float* x, float* y) {
	
	//preform multiplication using CSR format
	//loop over rows
	for(int i=0; i < num_row, i++) {
		float sum = 0;
		//loop over non-zero elements
		for(int i = row_ptr[i]; i < row_ptr[i + 1]; j++){
			sum += value[i] * x[col_idx[i]];
		}
		y[i] = sum;
	}
}

__global__ void spmvCuda(unsigned int num_row, const float* value, const int* col_idx, const int* row_ptr, const float* x, float* y){

	//calculate row to work on
	int row = blockDim.x * blockIdx.x + threadIdx.x;
	//If the row is within bounds preform multiplication for that row
	if(row < num_rows) {
		float sum = 0;
		for(int i = row_ptr[row]; i < row_ptr[row + 1]; j++){
			sum += value[i] * x[col_idx[i]];
		}
		y[row] = sum;
	}
}

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
	
	//declare up variables to read into
	unsigned int num_row, num_non_zero, num_col;
	float* value, x;
	int* col_idx, row_ptr;

	//first line of input is num_row+1
	getline(input, num_row);
	num_row--;

	//Second line of input is the number of non-zero elements (the number of elements in value)
       	getline(input, num_non_zero);

	//Third line is number of column in matrix / rows in vector
	getline(input, num_col);

	
	//allocate the variables for data
	row_ptr = (int*)malloc(sizeof(int) * (num_row+1));
	col_idx = (int *)malloc(sizeof(int) * num_col);
	value = (float *)malloc(sizeof(float) * num_non_zero);
	x = (float *)malloc(sizeof(float) * num_col);

	//read in data
	for(int i = 0; i < num_row + 1, i++){
		getline(input, row_ptr[i]);
	}
	for(int i = 0; i < num_col; i++){
		getline(input, col_idx[i]);
	}
	for(int i = 0; i < num_non_zero; i++){
		getline(input, value[i]);
	}
	for(int i = 0; i < num_col; i++){
		getline(input, x[i]);
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



	
	return 0;


}
