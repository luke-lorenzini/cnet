#include <stdio.h>
#include "utilities.h"

void print(Matrix_t* mat) {
	//printf("%c\n", inMat->ID);
	for (int i = 0; i < mat->Rows; i++) {
		for (int j = 0; j < mat->Cols; j++) {
			printf("%f ", *(mat->Matrix + i * mat->Cols + j));
		}
		printf("\n");
	}

	printf("\n");
}

void kill_memory(Matrix_t* p) {
	free(p->Matrix);
}