#include "cblas.h"
#include "wrap_blas.h"

const float alpha = 1;
const float beta = 1;
const int lda = 1;
const int ldb = 1;
const int ldc = 1;
const int incX = 1;
const int incY = 1;

void gemv(Matrix_t* a, Matrix_t* b, Matrix_t* c) {
	cblas_dgemv(CblasRowMajor, CblasNoTrans, a->Rows, a->Cols, alpha, a->Matrix, lda, b->Matrix, incX, beta, c->Matrix, incY);
}

void gemm(Matrix_t* a, Matrix_t* b, Matrix_t* c) {
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, a->Rows, a->Cols, b->Cols, alpha, a->Matrix, lda, b->Matrix, ldb, beta, c->Matrix, ldc);
}

void axpy(const double alpha, Matrix_t* x, Matrix_t* y) {
	cblas_daxpy(x->Rows, alpha, x->Matrix, incX, y->Matrix, incY);
}

void scal(const double scale, Matrix_t* mat) {
	cblas_dscal(mat->Rows, scale, mat->Matrix, incX);
}

void scopy(Matrix_t* x, Matrix_t* y) {
	cblas_dcopy(x->Rows, x->Matrix, incX, y->Matrix, incY);
}

float snorm(Matrix_t* mat) {
	return cblas_snrm2(mat->Rows, mat->Matrix, incX);
}

double dnorm(Matrix_t* mat) {
	return cblas_dnrm2(mat->Rows, mat->Matrix, incX);
}
