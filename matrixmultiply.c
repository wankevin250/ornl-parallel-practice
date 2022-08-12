#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

int normalMatrixMultiply(int matARow, int matACol, int matBRow, int matBCol, double matrixA[matARow][matACol], double matrixB[matBRow][matBCol]);
int ompMatrixMultiply(int matARow, int matACol, int matBRow, int matBCol, double matrixA[matARow][matACol], double matrixB[matBRow][matBCol]);
int ompOffloadingMatrixMultiply(int matARow, int matACol, int matBRow, int matBCol, double matrixA[matARow][matACol], double matrixB[matBRow][matBCol]);


int main(int argc, char** argv) {
    double start;
    double end;
    int matARow, matACol, matBRow, matBCol;
    int i, j, k;

    srand(time(0));
    
    setbuf(stdout, NULL);
    printf("Hello!\n");

    printf("Please enter the number of rows of matrix A\n");
    scanf("%d", &matARow);

    printf("Please enter the number of columns of matrix A\n");
    scanf("%d", &matACol);

    printf("Please enter the number of rows of matrix B\n");
    scanf("%d", &matBRow);

    printf("Please enter the number of columns of matrix B\n");
    scanf("%d", &matBCol);

    double (*matrixA)[matACol] = malloc(sizeof(double[matARow][matACol]));
    double (*matrixB)[matBCol] = malloc(sizeof(double[matBRow][matBCol]));

    if (matACol != matBRow) {
        printf("Dimensions incompatible for matrix multiplication\n");
        return 1;
    }

//Parallel way to enter random values for all values of matrices A and B
#pragma omp parallel default(none) shared(matARow, matACol, matBRow, matBCol, matrixA, matrixB) private(i,j,k)
    {
	#pragma omp for schedule(guided) collapse(2)
            for (i = 0; i < matARow; i++) {
                for (j = 0; j < matACol; j++) {
                    //printf("Please enter the element at row %d, column %d of matrix A\n", i, j);
                    //scanf("%d", &(matrixA[i][j]));
                    matrixA[i][j] = rand(); //CHANGE ARRAY TO DOUBLE
                }
            }
    
        #pragma omp for schedule(guided) collapse(2)
            for (i = 0; i < matBRow; i++) {
                for (j = 0; j < matBCol; j++) {
                    //printf("Please enter the element at row %d, column %d of matrix B\n", i, j);
                    //scanf("%d", &(matrixB[i][j]));
                    matrixB[i][j] = rand();
                }
            }
    }

    //Recording the time before and after each function to record the time each takes
    start = omp_get_wtime();    
    normalMatrixMultiply(matARow, matACol, matBRow, matBCol, matrixA, matrixB);
    end = omp_get_wtime();
    printf("Normal matrix work took %lf seconds\n", end - start);

    start = omp_get_wtime();
    ompMatrixMultiply(matARow, matACol, matBRow, matBCol, matrixA, matrixB);
    end = omp_get_wtime();
    printf("Simple OMP Threading work took %lf seconds\n", end - start);

    start = omp_get_wtime();
    ompOffloadingMatrixMultiply(matARow, matACol, matBRow, matBCol, matrixA, matrixB);
    end = omp_get_wtime();
    printf("OpenMP GPU Offloading work took %lf seconds\n", end - start);

    free(matrixA);
    free(matrixB);

    return 0;
}

//Function to perform matrix multiplication without any form of attempted speed-ups
int normalMatrixMultiply(int matARow, int matACol, int matBRow, int matBCol, double matrixA[matARow][matACol], double matrixB[matBRow][matBCol]) {
    int i;
    int j;
    int k;

    double (*matrixC)[matBCol] = malloc(sizeof(double[matARow][matBCol]));
    for (i = 0; i < matARow; i++) {
        for (j = 0; j < matBCol; j++) {
            for (k = 0; k < matACol; k++) {
                matrixC[i][j] += matrixA[i][k] * matrixB[k][j];
		
	    }
        }
    }

    //for (i = 0; i < matARow; i++) {
    //    for (j = 0; j < matBCol; j++) {
    //            if (j == matBCol - 1) {
    //		    printf("%d\n", matrixC[i][j]);
    //		} else {
    //              printf("%d ", matrixC[i][j]);
    //		}
    //	}
    //}

    free(matrixC);

    return 0; 
}

//Function to perform matrix multiplicatiom under OMP parallel threading
int ompMatrixMultiply(int matARow, int matACol, int matBRow, int matBCol, double matrixA[matARow][matACol], double matrixB[matBRow][matBCol]) {
    int i;
    int j;
    int k;


    double (*matrixC)[matBCol] = malloc(sizeof(double[matARow][matBCol]));

#pragma omp parallel default(none) shared(matARow, matBCol, matACol, matrixA, matrixB, matrixC) private(i,j,k)
{
        #pragma omp for schedule(guided) collapse(2)
            for (i = 0; i < matARow; i++) {
                for (j = 0; j < matBCol; j++) {
                    for (k = 0; k < matACol; k++) {
                        matrixC[i][j] += matrixA[i][k] * matrixB[k][j];	
	            }
                }
            }

            //for (i = 0; i < matARow; i++) {
            //    for (j = 0; j < matBCol; j++) {
            //        if (j == matBCol - 1) {
	    //        printf("%d\n", matrixC[i][j]);
	    //	    } else {
            //          printf("%d ", matrixC[i][j]);
	    //	    }
	    // }
            //}
}

    free(matrixC);

    return 0; 
}

//Function to perform matrix multiplication with OMP GPU Offloading
int ompOffloadingMatrixMultiply(int matARow, int matACol, int matBRow, int matBCol, double matrixA[matARow][matACol], double matrixB[matBRow][matBCol]) {
    int i;
    int j;
    int k;
    double start, end;

//include time for just the calculation and the time for the whole thing
#pragma omp target enter data map (to: matrixA[0:matARow][0:matACol], matrixB[0:matBRow][0:matBCol])
start = omp_get_wtime();

double (*matrixC)[matBCol] = malloc(sizeof(double[matARow][matBCol]));

#pragma omp target map(to: matrixA, matrixB) map(tofrom: matrixC)
{
    #pragma omp teams distribute parallel for simd collapse(2) num_teams(16)
            for (i = 0; i < matARow; i++) {
                for (j = 0; j < matBCol; j++) {
                    for (k = 0; k < matACol; k++) {
                        matrixC[i][j] += matrixA[i][k] * matrixB[k][j];	
	            }
                }
            }
}
            //for (i = 0; i < matARow; i++) {
            //    for (j = 0; j < matBCol; j++) {
            //        if (j == matBCol - 1) {
	    //        printf("%d\n", matrixC[i][j]);
	    //	    } else {
            //          printf("%d ", matrixC[i][j]);
	    //	    }
	    // }
            //}

    free(matrixC);
    end = omp_get_wtime();
    printf("OpenMP GPU Offloading time without allocation time is %lf\n", end - start);
    return 0; 

}
