#ifndef WRAP_CBLAS_H
#define WRAP_CBLAS_H

#include "cblas.h"

//#define USE_CUDA

typedef struct {
	int Rows;
	int Cols;
	double* Matrix;
	char ID;
} Matrix_t;

void axpy(const double, Matrix_t* x, Matrix_t* y);
void gemv(Matrix_t* a, Matrix_t* b, Matrix_t* c);
void gemm(Matrix_t* a, Matrix_t* b, Matrix_t* c);
void scal(const double scale, Matrix_t* mat);
void dcopy(Matrix_t* x, Matrix_t* y);
double dnorm(Matrix_t* mat);
void init_blas(void);
void deinit_blas(void);

#ifdef __cplusplus
}
#endif

#endif
