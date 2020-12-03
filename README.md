# CS4370-Project4
This is a cuda program that covers sparce matrix-vector multiplication for class.

## Editing BLOCKSIZE
* A define statement for BLOCKSIZE can be found on line 9 of the .cu file


## Compiling
nvcc was used to compile these programs. This will create an executable program.
* Command for compiling sum reduction: nvcc flemingSPMV.cu -o SPMV

## Running
These programs can be run directly from the command line.
* Command for parallel sum reduction: {path to executable}/SPMV {path to data file}
