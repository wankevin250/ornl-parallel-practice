#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

int normalMatrixMultiply(int matARow, int matACol, int matBRow, int matBCol, int matrixA[matARow][matACol], int matrixB[matBRow][matBCol]);
int ompMatrixMultiply(int matARow, int matACol, int matBRow, int matBCol, int matrixA[matARow][matACol], int matrixB[matBRow][matBCol]);
int ompOffloadingMatrixMultiply(int matARow, int matACol, int matBRow, int matBCol, int matrixA[matARow][matACol], int matrixB[matBRow][matBCol]);


int main(int argc, char** argv) {
    double start;
    double end;
    int matARow, matACol, matBRow, matBCol;

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

    int (*matrixA)[matACol] = malloc(sizeof(int[matARow][matACol]));
    int (*matrixB)[matBCol] = malloc(sizeof(int[matBRow][matBCol]));

    if (matACol != matBRow) {
        printf("Dimensions incompatible for matrix multiplication\n");
        return 1;
    }

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

int normalMatrixMultiply(int matARow, int matACol, int matBRow, int matBCol, int matrixA[matARow][matACol], int matrixB[matBRow][matBCol]) {
    int i;
    int j;
    int k;

    for (i = 0; i < matARow; i++) {
        for (j = 0; j < matACol; j++) {
            //printf("Please enter the element at row %d, column %d of matrix A\n", i, j);
            //scanf("%d", &(matrixA[i][j]));
            matrixA[i][j] = rand();
        }
    }

    for (i = 0; i < matBRow; i++) {
        for (j = 0; j < matBCol; j++) {
            //printf("Please enter the element at row %d, column %d of matrix B\n", i, j);
            //scanf("%d", &(matrixB[i][j]));
            matrixB[i][j] = rand();
        }
    }

    int (*matrixC)[matBCol] = malloc(sizeof(int[matARow][matBCol]));
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

int ompMatrixMultiply(int matARow, int matACol, int matBRow, int matBCol, int matrixA[matARow][matACol], int matrixB[matBRow][matBCol]) {
    int i;
    int j;
    int k;

#pragma omp parallel default(none) shared(matARow, matACol, matBRow, matBCol, matrixA, matrixB) private(i,j,k)
    {
	#pragma omp for schedule(guided) collapse(2)
            for (i = 0; i < matARow; i++) {
                for (j = 0; j < matACol; j++) {
                    //printf("Please enter the element at row %d, column %d of matrix A\n", i, j);
                    //scanf("%d", &(matrixA[i][j]));
                    matrixA[i][j] = rand();
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

    int (*matrixC)[matBCol] = malloc(sizeof(int[matARow][matBCol]));

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

int ompOffloadingMatrixMultiply(int matARow, int matACol, int matBRow, int matBCol, int matrixA[matARow][matACol], int matrixB[matBRow][matBCol]) {
    int i;
    int j;
    int k;
    double start, end;

//include time for just the calculation and the time for the whole thing
#pragma omp target enter data map (to: matrixA[0:matARow][0:matACol], matrixB[0:matBRow][0:matBCol])
start = omp_get_wtime();
#pragma omp target map(tofrom: matrixA)
{
    #pragma omp teams distribute parallel for simd num_teams(16) collapse(2)
    for (i = 0; i < matARow; i++) {
        for (j = 0; j < matACol; j++) {
            //printf("Please enter the element at row %d, column %d of matrix A\n", i, j);
            //scanf("%d", &(matrixA[i][j]));
            matrixA[i][j] = rand();
        }
    }
}

#pragma omp target map(tofrom: matrixB)
{
    #pragma omp teams distribute parallel for simd num_teams(16) collapse(2)
            for (i = 0; i < matBRow; i++) {
                for (j = 0; j < matBCol; j++) {
                    //printf("Please enter the element at row %d, column %d of matrix B\n", i, j);
                    //scanf("%d", &(matrixB[i][j]));
                    matrixB[i][j] = rand();
                }
            }
}

    int (*matrixC)[matBCol] = malloc(sizeof(int[matARow][matBCol]));

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

//int hipMatrixMultiply(int matARow, int matACol, int matBRow, int matBCol, int matrixA[matARow][matACol], int matrixB[matBRow][matBCol]) {
//
//}
