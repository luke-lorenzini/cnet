#include "cblas.h"
#include "wrap_blas.h"

#ifdef USE_CUDA
#include "cublas_v2.h"
#include "cuda_runtime.h"
#endif

const float alpha = 1;
const float beta = 1;
const int lda = 1;
const int ldb = 1;
const int ldc = 1;
const int incX = 1;
const int incY = 1;

#ifdef USE_CUDA
cublasHandle_t handle;
#endif

void init_blas() {
#ifdef USE_CUDA
	cublasCreate_v2(&handle);
#endif
}

void deinit_blas() {
#ifdef USE_CUDA
	cublasDestroy(handle);
#endif
}

void gemv(Matrix_t* a, Matrix_t* b, Matrix_t* c) {
	cblas_dgemv(CblasRowMajor, CblasNoTrans, a->Rows, a->Cols, alpha, a->Matrix, lda, b->Matrix, incX, beta, c->Matrix, incY);
}

void gemm(Matrix_t* a, Matrix_t* b, Matrix_t* c) {
#ifdef USE_CUDA
	double* GPU_A, * GPU_B, * GPU_C;
	cudaMalloc(&GPU_A, (size_t)a->Rows * a->Cols * sizeof(double));
	cudaMalloc(&GPU_B, (size_t)b->Rows * b->Cols * sizeof(double));
	cudaMalloc(&GPU_C, (size_t)c->Rows * c->Cols * sizeof(double));

	cudaMemcpy(GPU_A, a->Matrix, (size_t)a->Rows * a->Cols * sizeof(double), cudaMemcpyDefault);
	cudaMemcpy(GPU_B, b->Matrix, (size_t)b->Rows * b->Cols * sizeof(double), cudaMemcpyDefault);
	cudaMemcpy(GPU_C, c->Matrix, (size_t)c->Rows * c->Cols * sizeof(double), cudaMemcpyDefault);

	const double alf = 1;
	const double bet = 0;
	const double* alpha = &alf;
	const double* beta = &bet;

	int lda = a->Cols;
	int ldb = a->Cols;
	int ldc = a->Rows;

	cublasDgemm(handle,
		CUBLAS_OP_T, CUBLAS_OP_N,
		a->Rows, b->Cols, a->Cols,
		alpha,
		GPU_A, lda,
		GPU_B, ldb,
		beta,
		GPU_C, ldc);

	cudaMemcpy(c->Matrix, GPU_C, (size_t)c->Rows * c->Cols * sizeof(double), cudaMemcpyDefault);

	cudaFree(GPU_A);
	cudaFree(GPU_B);
	cudaFree(GPU_C);
#else
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, a->Rows, a->Cols, b->Cols, alpha, a->Matrix, lda, b->Matrix, ldb, beta, c->Matrix, ldc);
#endif
}

void axpy(const double alpha, Matrix_t* x, Matrix_t* y) {
	cblas_daxpy(x->Rows, alpha, x->Matrix, incX, y->Matrix, incY);
}

void scal(const double scale, Matrix_t* mat) {
#ifdef USE_CUDA
	double* GPU_A;
	cudaMalloc(&GPU_A, (size_t)mat->Rows * mat->Cols * sizeof(double));

	cudaMemcpy(GPU_A, mat->Matrix, (size_t)mat->Rows * mat->Cols * sizeof(double), cudaMemcpyDefault);

	const double* alpha = &scale;

	cublasDscal(handle, mat->Rows,
		alpha,
		GPU_A, incX);

	cudaMemcpy(mat->Matrix, GPU_A, (size_t)mat->Rows * mat->Cols * sizeof(double), cudaMemcpyDefault);

	cudaFree(GPU_A);
#else
	cblas_dscal(mat->Rows, scale, mat->Matrix, incX);
#endif
}

void scopy(Matrix_t* x, Matrix_t* y) {
	cblas_dcopy(x->Rows, x->Matrix, incX, y->Matrix, incY);
}

double dnorm(Matrix_t* mat) {
	return cblas_dnrm2(mat->Rows, mat->Matrix, incX);
}
