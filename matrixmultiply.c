#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    int i;
    int j;
    int k;
    int matARow, matACol, matBRow, matBCol;
    
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

    for (i = 0; i < matARow; i++) {
        for (j = 0; j < matACol; j++) {
            printf("Please enter the element at row %d, column %d of matrix A\n", i, j);
            scanf("%d", &(matrixA[i][j]));
        }
    }

    for (i = 0; i < matBRow; i++) {
        for (j = 0; j < matBCol; j++) {
            printf("Please enter the element at row %d, column %d of matrix B\n", i, j);
            scanf("%d", &(matrixB[i][j]));
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

    for (i = 0; i < matARow; i++) {
        for (j = 0; j < matBCol; j++) {
                if (j == matBCol - 1) {
		    printf("%d\n", matrixC[i][j]);
		} else {
                    printf("%d ", matrixC[i][j]);
		}
	}
    }

    free(matrixA);
    free(matrixB);
    free(matrixC);

    return 0;   
}
