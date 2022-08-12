#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include <hip/hip_runtime.h>
#include <iostream>
#include <hiprand/hiprand.h>
#include <hiprand/hiprand_kernel.h>

#define HIP_CHECK(command) {                                                              \
    hipError_t status = command;                                                          \
                                                                                          \
    if (status != hipSuccess) {                                                           \
        std::cerr << "Error: HIP reports " << hipGetErrorString(status) << std::endl;     \
                                                                                          \
        std::abort();                                                                     \
    } }                                                                  

//GPU Kernel to generate random values for each cell of the 2D matrix, in 1D form        
__global__  void fillMatrixKernel(int row, int col, int *matrix_d) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int seed = threadIdx.x;

    hiprandState s;

    hiprand_init(seed, 0, 0, &s);

    if (i < row * col) {
        //matrix_d[i] = hiprandGenerate(generator, matrix_d[i]);
        matrix_d[i] = hiprand(&s);
    }
}

//GPU Kernel to conduct the actual multiplication of matrices together
__global__ void multiplicationKernel(double* matA, double* matB, double* matC, int matARow, int matACol, int matBRow, int matBCol, size_t arrDim) {
    size_t offsetThread = threadIdx.x + blockIdx.x * blockDim.x; // Thread number
    size_t totalThreads = gridDim.x * blockDim.x; //Bus number
    double tmp;

    tmp = 0;

    for (size_t i = offsetThread; i < arrDim; i += totalThreads) {
        
        size_t c_row = i / matBCol;
        size_t c_col = i & matBCol;
        matC[i] = 0;

        size_t indexA = c_row * matACol;
        size_t indexB = c_col;

        for (size_t j = 0; j < matACol; j++) {
            tmp += matA[indexA] * matB[indexB];
            indexA++;
            indexB += matBCol;
        }
        matC[i] = tmp;
    }
}

int main(int argc, char** argv) {
    int i, j, k;
    int matARow, matACol, matBRow, matBCol;
    double* matrixA;
    double* matrixB;
    double* matrixC;
    double* matrixA_d;
    double* matrixB_d;
    double* matrixC_d;

    //Takes in the dimensions of the matrices to be multiplied
    setbuf(stdout, NULL);
    std::cout << "Please enter the number of rows of matrix A." << std::endl;
    std::cin >> matARow;

    std::cout << "Please enter the number of columns of matrix A." << std::endl;
    std::cin >> matACol;

    std::cout << "Please enter the number of rows of matrix B." << std::endl;
    std::cin >> matBRow;

    std::cout << "Please enter the number of columns of matrix B." << std::endl;
    std::cin >> matBCol;

    if (matACol != matBRow) {
        std::cout << "The dimensions of the two arrays are impossible to multiply!" << std::endl;
        return 1;
    }

    srand(time(0));

    matrixA = (double*) malloc(sizeof(double)* matARow * matACol);
    matrixB = (double*) malloc(sizeof(double)* matBRow * matBCol);
    matrixC = (double*) malloc(sizeof(double)* matARow * matBCol);

//CPU version of randomly generating values for the matrices in parallel
#pragma omp parallel default(none) private(i,j) shared(matrixA, matrixB, matARow, matACol, matBRow, matBCol)
{
    #pragma omp for collapse(2)
    for (i = 0; i < matARow; i++) {
        for (j = 0; j < matACol; j++) {
            matrixA[i + j * matACol] = rand();
        }
    }

    #pragma omp for collapse(2)
    for (i = 0; i < matBRow; i++) {
        for (j = 0; j < matBCol; j++) {
            matrixB[i + j * matBCol] = rand();
        }
    }        
}

    //srand(time(0));
    //hiprandGenerator_t generator;
    //hiprandCreateGenerator(&generator, HIPRAND_RNG_QUASI_DEFAULT);
    //clock_t start_time = clock();

    int numDevices = 0;
    HIP_CHECK(hipGetDeviceCount(&numDevices));

    //hipStream_t stream1;
    //HIP_CHECK(hipStreamCreate(&stream1));

    //hipStream_t stream2;
    //HIP_CHECK(hipStreamCreate(&stream2));

    HIP_CHECK(hipMalloc(&matrixA_d, sizeof(double) * matARow * matACol));
    HIP_CHECK(hipMalloc(&matrixB_d, sizeof(double) * matBRow * matBCol));

    //Calling the kernel to copy over matrixA to the device matrixA_d
    HIP_CHECK(hipMemcpy(matrixA_d, matrixA, sizeof(double) * matARow * matACol, hipMemcpyHostToDevice));

    //Calling the kernel to copy over matrixB to the device matrixB_d
    HIP_CHECK(hipMemcpy(matrixB_d, matrixB, sizeof(double) * matBRow * matBCol, hipMemcpyHostToDevice));

    //Calling the kernel to fill up matrixA and matrixB with random values
    //hipLaunchKernelGGL(fillMatrixKernel, dim3(1), dim3(256), 0, stream1, matARow, matACol, matrixA_d);
    
    //hipLaunchKernelGGL(fillMatrixKernel, dim3(1), dim3(256), 0, stream2, matBRow, matBCol, matrixB_d);

    //Copying the memory from the device back to the host for matrixA and matrixB
    //HIP_CHECK(hipMemcpy(matrixA, matrixA_d, sizeof(int) * matARow * matACol, hipMemcpyDeviceToHost));

    //HIP_CHECK(hipMemcpy(matrixB, matrixB_d, sizeof(int) * matBRow * matBCol, hipMemcpyDeviceToHost));

    //HIP_CHECK(hipStreamDestroy(stream1));
    //HIP_CHECK(hipStreamDestroy(stream2));

    //Allocating the space on the device for matrixC
    HIP_CHECK(hipMalloc(&matrixC_d, sizeof(double) * matARow * matBCol));

    clock_t start_time = clock();

    hipLaunchKernelGGL(multiplicationKernel, dim3(1), dim3(256), 0, 0, matrixA_d, matrixB_d, matrixC_d, matARow, matACol, matBRow, matBCol, matARow * matBCol);

    HIP_CHECK(hipMemcpy(matrixC, matrixC_d, sizeof(double) * matARow * matBCol, hipMemcpyDeviceToHost));

    clock_t end_time = clock();

    //for (i = 0; i < matARow; i++) {
    //    for (j = 0; j < matBCol; j++) {
    //        std::cout << *(matrixC + i*matBCol + j) << ' ';
    //    }
    //    std::cout << std::endl;
    //}

    std::cout << "Total elapsed time for HIP matrix multiplication is: " << ((double) (end_time - start_time)) / CLOCKS_PER_SEC << std::endl;

    HIP_CHECK(hipFree(matrixA_d));
    HIP_CHECK(hipFree(matrixB_d));
    HIP_CHECK(hipFree(matrixC_d));
    
    free(matrixA);
    free(matrixB);
    free(matrixC);

    //hiprandDestroyGenerator(generator);

    return 0;
}

