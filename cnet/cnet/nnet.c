#include <math.h>
#include <stdlib.h>
#include <time.h>

#include "data.h"
#include "import_data.h"
#include "nnet.h"
#include "utilities.h"

#ifdef USE_CUDA
#include "cublas_v2.h"
#include "cuda_runtime.h"
#endif

Matrix_t scal_add(double val, Matrix_t* source);
void init_W(Matrix_t* mat, int size);
void trans(Matrix_t* A, Matrix_t* B);
void calc_drelu(Matrix_t* vecIn, Matrix_t* vecOut);
void calc_relu(Matrix_t* vecIn, Matrix_t* vecOut);
void calc_tanh(Matrix_t* vecIn, Matrix_t* vecOut);
void calc_dtanh(Matrix_t* vecIn, Matrix_t* vecOut);
void calc_sigmoid(Matrix_t* vecIn, Matrix_t* vecOut);
void calc_softmax(Matrix_t* vecIn, Matrix_t* vecOut);
void mult(Matrix_t* x, Matrix_t* y, Matrix_t* out);
void add(Matrix_t* x, Matrix_t* y, Matrix_t* out);
double find_max(Matrix_t* vecIn);
void calc_log(Matrix_t* inMat, Matrix_t* outMat);

Config_t network[4] = {
	{
		ROWS_0,
		COLS_0
	},
	{
		ROWS_1,
		COLS_1
	},
	{
		ROWS_2,
		COLS_2
	},
	{
		ROWS_3,
		COLS_3
	}
};

// Backprop vars
Matrix_t Wt[LAYERS];
Matrix_t zTemp[LAYERS];
Matrix_t at[LAYERS];
Matrix_t temp0[LAYERS];
Matrix_t temp1[LAYERS];
Matrix_t tempy;

clock_t startLoss, endLoss, totalLoss;

// cost
// J += -(y log(a[N]) + (1 - y) log(1 - a[N]))
void calculate_loss(Matrix_t* J, Matrix_t* ptr_a, Matrix_t* ptr_y) {
	startLoss = clock();

	Matrix_t ones;
	Matrix_t temp3;
	Matrix_t temp4;
	Matrix_t temp5;
	Matrix_t A;
	Matrix_t B;

	ones.Rows = ptr_y->Rows;
	ones.Cols = VECTOR_WIDTH;
	ones.Matrix = (double*)calloc((size_t)ones.Rows * ones.Cols, sizeof(double));
	for (int i = 0; i < ones.Rows; i++) {
		ones.Matrix[i] = 1;
	}
	temp3.Rows = ptr_y->Rows;
	temp3.Cols = VECTOR_WIDTH;
	temp3.Matrix = (double*)calloc((size_t)temp3.Rows * temp3.Cols, sizeof(double));
	temp4.Rows = ptr_y->Rows;
	temp4.Cols = VECTOR_WIDTH;
	temp4.Matrix = (double*)calloc((size_t)temp4.Rows * temp4.Cols, sizeof(double));
	temp5.Rows = ptr_y->Rows;
	temp5.Cols = VECTOR_WIDTH;
	temp5.Matrix = (double*)calloc((size_t)temp5.Rows * temp5.Cols, sizeof(double));
	A.Rows = ptr_y->Rows;
	A.Cols = VECTOR_WIDTH;
	A.Matrix = (double*)calloc((size_t)A.Rows * A.Cols, sizeof(double));
	B.Rows = ptr_y->Rows;
	B.Cols = VECTOR_WIDTH;
	B.Matrix = (double*)calloc((size_t)B.Rows * B.Cols, sizeof(double));

	// Calc A
	calc_log(ptr_a, &temp3);
	mult(ptr_y, &temp3, &A);

	// Calc B1
	scopy(ptr_y, &temp4);
	scal(-1, &temp4);
	axpy(1, &ones, &temp4);

	// Calc B2
	scopy(ptr_a, &temp5);
	scal(-1, &temp5);
	axpy(1, &ones, &temp5);
	calc_log(&temp5, &temp5);

	// Calc J
	// todo j should be 1 number, not 3
	mult(&temp4, &temp5, &temp5);
	axpy(1, &A, &temp5);
	axpy(-1, &temp5, J);

	kill_memory(&ones);
	kill_memory(&temp3);
	kill_memory(&temp4);
	kill_memory(&temp5);
	kill_memory(&A);
	kill_memory(&B);

	endLoss = clock();
	totalLoss += (endLoss - startLoss);
}

double getLossTime() {
	return totalLoss;
}

clock_t startFwd, endFwd, totalFwd;

// fwd prop
// z[n] = (W[n] DOT a[n - 1]) + b[n]
// a[n] = relu(z[n])
// a[N] = sig(z[N])
void fwd_prop(Matrix_t* W, Matrix_t* b, Matrix_t* a, Matrix_t* z) {
	startFwd = clock();

	for (int layer = 1; layer < LAYERS; layer++) {
		// z = Wx + b
		//printf("W[%d]\n", layer);
		//print(&W[layer]);
		//printf("a[%d]\n", layer-1);
		//print(&a[layer-1]);
		gemm(&W[layer], &a[layer - 1], &z[layer]);
		//printf("z[%d]\n", layer);
		//print(&z[layer]);

		axpy(1, &b[layer], &z[layer]);

		if (layer != (LAYERS - 1)) {
			calc_relu(&z[layer], &a[layer]);
		}
		else {
			if (z[layer].Rows == 1) {
				calc_sigmoid(&z[layer], &a[layer]);
			}
			else {
				calc_softmax(&z[layer], &a[layer]);
			}
		}
	}

	endFwd = clock();
	totalFwd += (endFwd - startFwd);
}

double getFwdTime() {
	return totalFwd;
}

clock_t startBkwd, endBkwd, totalBkwd;

// back prop
// dz[N] = a[N] - y
// dz[n] = WT[n + 1] DOT dz[n + 1] * drelu(z[n])
// db[n] += dz[n]
// dW[n] += dz[n] DOT aT[n - 1]
void back_prop(Matrix_t* W, Matrix_t* b, Matrix_t* z, Matrix_t* a, Matrix_t* y, Matrix_t* dW, Matrix_t* db, Matrix_t* dz) {
	startBkwd = clock();

	//Matrix_t Wt[LAYERS];
	//Matrix_t zTemp[LAYERS];
	//Matrix_t at[LAYERS];
	//Matrix_t temp0[LAYERS];
	//Matrix_t temp1[LAYERS];
	//Matrix_t tempy;

	tempy.Rows = y->Rows;
	tempy.Cols = y->Cols;
	tempy.Matrix = (double*)calloc((size_t)tempy.Rows * tempy.Cols, sizeof(double));

	//for (int idx = 0; idx < LAYERS; idx++) {
	//	Wt[idx].Rows = network[idx].Cols;
	//	Wt[idx].Cols = network[idx].Rows;
	//	Wt[idx].Matrix = (double*)calloc(Wt[idx].Rows * Wt[idx].Cols, sizeof(double));

	//	zTemp[idx].Rows = network[idx].Rows;
	//	zTemp[idx].Cols = VECTOR_WIDTH;
	//	zTemp[idx].Matrix = (double*)calloc(zTemp[idx].Rows * zTemp[idx].Cols, sizeof(double));

	//	at[idx].Rows = VECTOR_WIDTH;
	//	at[idx].Cols = network[idx].Rows;
	//	at[idx].Matrix = (double*)calloc(at[idx].Rows * at[idx].Cols, sizeof(double));

	//	temp0[idx].Rows = network[idx].Rows;
	//	temp0[idx].Cols = network[idx].Cols;
	//	temp0[idx].Matrix = (double*)calloc(temp0[idx].Rows * temp0[idx].Cols, sizeof(double));

	//	temp1[idx].Rows = network[idx].Rows;
	//	temp1[idx].Cols = VECTOR_WIDTH;
	//	temp1[idx].Matrix = (double*)calloc(temp1[idx].Rows * temp1[idx].Cols, sizeof(double));
	//}

	for (int layer = LAYERS - 1; layer > 0; layer--) {
		if (layer == (LAYERS - 1)) {
			// dz[N]
			scopy(y, &tempy);
			scal(-1, &tempy);
			axpy(1, &a[layer], &tempy);
			scopy(&tempy, &dz[layer]);
		}
		else {
			// dz[n]
			trans(&W[layer + 1], &Wt[layer + 1]);
			gemm(&Wt[layer + 1], &dz[layer + 1], &temp1[layer]);
			calc_drelu(&z[layer], &zTemp[layer]);
			mult(&temp1[layer], &zTemp[layer], &dz[layer]);
		}

		// db[n]
		axpy(1, &dz[layer], &db[layer]); 

		// dW[n]
		trans(&a[layer - 1], &at[layer - 1]);
		gemm(&dz[layer], &at[layer - 1], &temp0[layer]);
		add(&temp0[layer], &dW[layer], &dW[layer]);
	}

	//for (int layer = 0; layer < LAYERS; layer++) {
	//	kill_memory(&Wt[layer]);
	//	kill_memory(&zTemp[layer]);
	//	kill_memory(&at[layer]);
	//	kill_memory(&temp0[layer]);
	//	kill_memory(&temp1[layer]);
	//}
	kill_memory(&tempy);

	endBkwd = clock();
	totalBkwd += (endBkwd - startBkwd);
}

double getBkwdTime() {
	return totalBkwd;
}

void update_weights(Matrix_t* W, Matrix_t* b, Matrix_t* dW, Matrix_t* db, Matrix_t* J, int epoch) {
	// update
	// W[1] -= alpha * dW[1] / m
	// b[1] -= alpha * db[1] / m
	// J /= m
	double inverseSize = 1 / (double)RECORDS;
	// learning rate decay
	double learningRateInitial = 0.2;
	double decayRate = 1;
	decayRate = 0.02;
	double learningRate = learningRateInitial / (1 + decayRate * epoch);
	double scale = learningRate * inverseSize;

	for (int layer = 1; layer < LAYERS; layer++) {
		regularize(&W[layer], &dW[layer]);
		scal_mult(scale, &dW[layer], &dW[layer]);
		subtract(&W[layer], &dW[layer], &W[layer]);
		scal_mult(scale, &db[layer], &db[layer]);
		subtract(&b[layer], &db[layer], &b[layer]);
	}
	scal(inverseSize, J);

	if (epoch % (EPOCHS / 10) == 0) {
		for (int layer = 1; layer < LAYERS; layer++) {
			//printf("W[%d]\n", layer);
			//print(&W[layer]);
			//printf("b[%d]\n", layer);
			//print(&b[layer]);
		}
		//gradcheck(W, b, dW, db);
		printf("J[%d]\n", epoch);
		printf("Learning Rate %f\n", learningRate);
		printf("%f\n\n", sum_vector(J));
	}

	//reset j, dw, db
	for (int layer = 1; layer < LAYERS; layer++) {
		zeros(&dW[layer]);
		zeros(&db[layer]);
	}
	zeros(J);
}

void gradcheck(Matrix_t* W, Matrix_t* b, Matrix_t* dW, Matrix_t* db) {
	const int index = 2;
	Matrix_t gradCheck_W;
	Matrix_t gradCheck_Wdiff;
	Matrix_t gradCheck_dW;
	Matrix_t gradCheck_dWdiff;

	gradCheck_W.Rows = W[index].Rows * W[index].Cols + b[index].Rows * b[index].Cols;
	gradCheck_W.Cols = VECTOR_WIDTH;
	gradCheck_W.Matrix = (double*)calloc((size_t)gradCheck_W.Rows * gradCheck_W.Cols, sizeof(double));
	//memcpy(&gradCheck_W.Matrix, W[index].Matrix, W[index].Rows * W[index].Cols * sizeof(double));
	//memcpy(&gradCheck_W.Matrix[W[index].Rows * W[index].Cols], b[index].Matrix, b[index].Rows * b[index].Cols * sizeof(double));
	int count = 0;
	for (int i = 0; i < W[index].Rows * W[index].Cols; i++) {
		*(gradCheck_W.Matrix + i) = *(W[index].Matrix + i);
		count++;
	}
	for (int i = 0; i < b[index].Rows * b[index].Cols; i++) {
		*(gradCheck_W.Matrix + count) = *(b[index].Matrix + i);
		count++;
	}

	//printf("W[%d]\n", index);
	//print(&W[index]);
	//printf("b[%d]\n", index);
	//print(&b[index]);
	//printf("grad W\n");
	//print(&gradCheck_W);

	gradCheck_Wdiff.Rows = W[index].Rows * W[index].Cols + b[index].Rows * b[index].Cols;
	gradCheck_Wdiff.Cols = VECTOR_WIDTH;
	gradCheck_Wdiff.Matrix = (double*)calloc((size_t)gradCheck_Wdiff.Rows * gradCheck_Wdiff.Cols, sizeof(double));

	gradCheck_dW.Rows = dW[index].Rows * dW[index].Cols + db[index].Rows * db[index].Cols;
	gradCheck_dW.Cols = VECTOR_WIDTH;
	gradCheck_dW.Matrix = (double*)calloc((size_t)gradCheck_dW.Rows * gradCheck_dW.Cols, sizeof(double));
	//memcpy(&gradCheck_dW.Matrix, dW[index].Matrix, gradCheck_dW.Rows * gradCheck_dW.Cols * sizeof(double));
	//memcpy(&gradCheck_dW.Matrix[dW[index].Rows * dW[index].Cols], db[index].Matrix, db[index].Rows * db[index].Cols * sizeof(double));
	count = 0;
	for (int i = 0; i < dW[index].Rows * dW[index].Cols; i++) {
		gradCheck_dW.Matrix[i] = *(dW[index].Matrix + i);
		count++;
	}
	for (int i = 0; i < db[index].Rows * db[index].Cols; i++) {
		gradCheck_dW.Matrix[count] = *(db[index].Matrix + i);
		count++;
	}

	//printf("dW[%d]\n", index);
	//print(&dW[index]);
	//printf("db[%d]\n", index);
	//print(&db[index]);
	//printf("gradCheck_dW\n");
	//print(&gradCheck_dW);

	gradCheck_dWdiff.Rows = dW[index].Rows * dW[index].Cols + db[index].Rows * db[index].Cols;
	gradCheck_dWdiff.Cols = VECTOR_WIDTH;
	gradCheck_dWdiff.Matrix = (double*)calloc((size_t)gradCheck_dWdiff.Rows * gradCheck_dWdiff.Cols, sizeof(double));
			
	// grad  check
	double eps = 10E-7;
	Matrix_t res1;
	Matrix_t res2;
	res1 = scal_add(eps, &gradCheck_W);
	res2 = scal_add(-eps, &gradCheck_W);
	//printf("pRes1\n");
	//print(&res1);
	//printf("pRes2\n");
	//print(&res2);
	subtract(&res1, &res2, &gradCheck_Wdiff);
	//printf("apprx\n");
	//print(&gradCheck_Wdiff);
	scal((1/(2 * eps)), &gradCheck_Wdiff);
	//printf("apprx\n");
	//print(&gradCheck_Wdiff);
	double dThetaApprox = dnorm(&gradCheck_Wdiff);
	printf("norm(dThetaApprox) %f\n", dThetaApprox);

	double dTheta = dnorm(&gradCheck_dW);
	printf("norm(dTheta) %f\n", dTheta);

	subtract(&gradCheck_dW, &gradCheck_Wdiff, &gradCheck_dWdiff);
	//printf("network\n");
	//print(&gradCheck_dWdiff);
	double dThetaDiff = dnorm(&gradCheck_dWdiff);
	printf("norm(dThetaApprox - dTheta) %f\n", dThetaDiff);

	double grad = dThetaDiff / (dTheta + dThetaApprox);
	printf("grad %f\n", grad);
}

void regularize(Matrix_t* W, Matrix_t* dW) {
	double lambda = 0.4;
	double decayRate = lambda / (double)RECORDS;
	Matrix_t Wtemp;

	// dW = dW + lambda/m mult W
	//for (int layer = 1; layer < LAYERS; layer++) {
		Wtemp.Rows = W->Rows;
		Wtemp.Cols = W->Cols;
		Wtemp.Matrix = (double*)calloc((size_t)Wtemp.Rows * Wtemp.Cols, sizeof(double));

		scal_mult(decayRate, W, &Wtemp);
		add(dW, &Wtemp, dW);
	//}

		kill_memory(&Wtemp);
}

void deinit_network() {
#ifdef USE_CUDA
	deinit_blas();
#endif
	for (int layer = 0; layer < LAYERS; layer++) {
		kill_memory(&Wt[layer]);
		kill_memory(&zTemp[layer]);
		kill_memory(&at[layer]);
		kill_memory(&temp0[layer]);
		kill_memory(&temp1[layer]);
	}
}

void init_network(Matrix_t* W, Matrix_t* b, Matrix_t* x, Matrix_t* y, Matrix_t* a, Matrix_t* z, Matrix_t* dz, Matrix_t* dW, Matrix_t* db, Matrix_t* J) {
#ifdef USE_CUDA
	cudaError_t err;
	size_t size;

	init_blas();
#endif

	for (int i = 0; i < RECORDS; i++) {
		x[i].Rows = ROWS_0;
		x[i].Cols = VECTOR_WIDTH;
//#ifdef USE_CUDA
//		//__global__
//		size = x[i].Rows * x[i].Cols * sizeof(double);
//		err = cudaMallocManaged(&x[i].Matrix, size, cudaMemAttachGlobal);
//		zeros(&x[i]);
//#else
		x[i].Matrix = (double*)calloc((size_t)x[i].Rows * x[i].Cols, sizeof(double));
//#endif

		for (int j = 0; j < ROWS_0; j++) {
#ifndef USE_IMPORT
			x[i].Matrix[j] = IRIS_DATA[i][j];
#else
			x[i].Matrix[j] = get_logit(i, j);
#endif
		}
		//printf("x[%d]\n", i);
		//print(&x[i]);

		y[i].Rows = ROWS_3;
		y[i].Cols = VECTOR_WIDTH;
//#ifdef USE_CUDA
//		//__global__
//		size = y[i].Rows * y[i].Cols * sizeof(double);
//		err = cudaMallocManaged(&y[i].Matrix, size, cudaMemAttachGlobal);
//		zeros(&y[i]);
//#else
		y[i].Matrix = (double*)calloc((size_t)y[i].Rows * y[i].Cols, sizeof(double));
//#endif

		for (int j = 0; j < ROWS_3; j++) {
#ifndef USE_IMPORT
			if (ROWS_3 == 1) {
				if (i < 50) {
					y[i].Matrix[j] = 1;
				}
				else {
					y[i].Matrix[j] = 0;
				}
			}
			else {
				y[i].Matrix[j] = IRIS_LABEL[i][j];
			}
#else
			y[i].Matrix[j] = get_label(i, j);
#endif
		}
		//printf("y[%d]\n", i);
		//print(&y[i]);
	}

	J->Rows = ROWS_3;
	J->Cols = VECTOR_WIDTH;
	J->Matrix = (double*)calloc((size_t)J->Rows * J->Cols, sizeof(double));

	for (int idx = 1; idx < LAYERS; idx++) {
		W[idx].Rows = network[idx].Rows;
		W[idx].Cols = network[idx].Cols;
//#ifdef USE_CUDA
//		//__global__
//		size = W[idx].Rows * W[idx].Cols * sizeof(double);
//		err = cudaMallocManaged(&W[idx].Matrix, size, cudaMemAttachGlobal);
//		zeros(&W[idx]);
//#else
		W[idx].Matrix = (double*)calloc((size_t)W[idx].Rows * W[idx].Cols, sizeof(double));
//#endif
		init_W(&W[idx], W[idx].Cols);
		//printf("W[%d]\n", idx);
		//print(&W[idx]);

		b[idx].Rows = network[idx].Rows;
		b[idx].Cols = VECTOR_WIDTH;
		b[idx].Matrix = (double*)calloc((size_t)b[idx].Rows * b[idx].Cols, sizeof(double));
		//printf("b[%d]\n", idx);
		//print(&b[idx]);

		z[idx].Rows = network[idx].Rows;
		z[idx].Cols = VECTOR_WIDTH;
//#ifdef USE_CUDA
//		//__global__
//		size = z[idx].Rows * z[idx].Cols * sizeof(double);
//		err = cudaMallocManaged(&z[idx].Matrix, size, cudaMemAttachGlobal);
//		zeros(&z[idx]);
//#else
		z[idx].Matrix = (double*)calloc((size_t)z[idx].Rows * z[idx].Cols, sizeof(double));
//#endif

		a[idx].Rows = network[idx].Rows;
		a[idx].Cols = VECTOR_WIDTH;
		a[idx].Matrix = (double*)calloc((size_t)a[idx].Rows * a[idx].Cols, sizeof(double));

		dz[idx].Rows = network[idx].Rows;
		dz[idx].Cols = VECTOR_WIDTH;
		dz[idx].Matrix = (double*)calloc((size_t)dz[idx].Rows * dz[idx].Cols, sizeof(double));

		dW[idx].Rows = network[idx].Rows;
		dW[idx].Cols = network[idx].Cols;
		dW[idx].Matrix = (double*)calloc((size_t)dW[idx].Rows * dW[idx].Cols, sizeof(double));

		db[idx].Rows = network[idx].Rows;
		db[idx].Cols = VECTOR_WIDTH;
		db[idx].Matrix = (double*)calloc((size_t)db[idx].Rows * db[idx].Cols, sizeof(double));
	}

	for (int idx = 0; idx < LAYERS; idx++) {
		Wt[idx].Rows = network[idx].Cols;
		Wt[idx].Cols = network[idx].Rows;
		Wt[idx].Matrix = (double*)calloc((size_t)Wt[idx].Rows * Wt[idx].Cols, sizeof(double));

		zTemp[idx].Rows = network[idx].Rows;
		zTemp[idx].Cols = VECTOR_WIDTH;
		zTemp[idx].Matrix = (double*)calloc((size_t)zTemp[idx].Rows * zTemp[idx].Cols, sizeof(double));

		at[idx].Rows = VECTOR_WIDTH;
		at[idx].Cols = network[idx].Rows;
		at[idx].Matrix = (double*)calloc((size_t)at[idx].Rows * at[idx].Cols, sizeof(double));

		temp0[idx].Rows = network[idx].Rows;
		temp0[idx].Cols = network[idx].Cols;
		temp0[idx].Matrix = (double*)calloc((size_t)temp0[idx].Rows * temp0[idx].Cols, sizeof(double));

		temp1[idx].Rows = network[idx].Rows;
		temp1[idx].Cols = VECTOR_WIDTH;
		temp1[idx].Matrix = (double*)calloc((size_t)temp1[idx].Rows * temp1[idx].Cols, sizeof(double));
	}
}

Matrix_t scal_add(double val, Matrix_t* source) {
	Matrix_t sum;
	sum.Rows = source->Rows;
	sum.Cols = source->Cols;
	sum.Matrix = (double*)calloc((size_t)sum.Rows * sum.Cols, sizeof(double));

	for (int i = 0; i < source->Rows; i++) {
		sum.Matrix[i] = *(source->Matrix + i) + val;
	}

	return sum;
}

double rand_gen() {
	// return a uniformly distributed random value
	return ((double)(rand()) + 1.) / ((double)(RAND_MAX)+1.);
}

double normalRandom() {
	// return a normally distributed random value
	double v1 = rand_gen();
	double v2 = rand_gen();
	return cos(2 * 3.14 * v2) * sqrt(-2. * log(v1));
}

void init_W(Matrix_t* mat, int size) {
	double sigma = 82.0;
	double Mi = 40.0;
	sigma = 1;
	Mi = 1;
	double SCALE_FACTOR = sqrt(2 / (double)size);
	//SCALE_FACTOR = 0.1;
	//double sum = 0;
	//int count = 0;

	for (int row = 0; row < mat->Rows; row++) {
		for (int col = 0; col < mat->Cols; col++) {
			*(mat->Matrix + row * mat->Cols + col) = normalRandom() * SCALE_FACTOR * sigma * Mi;
			//*(mat->Matrix + row * mat->Cols + col) = rand() * SCALE_FACTOR;

			//sum += *(mat->Matrix + row * mat->Cols + col);
			//count++;
		}
	}

	//double mu = sum / count;

	//for (int row = 0; row < mat->Rows; row++) {
	//	for (int col = 0; col < mat->Cols; col++) {
	//		*(mat->Matrix + row * mat->Cols + col) -= mu;
	//	}
	//}
}

void trans(Matrix_t* A, Matrix_t* B) {
	for (int i = 0; i < A->Rows; i++) {
		for (int j = 0; j < A->Cols; j++) {
			// B[i][j] = A[j][i]
			*(B->Matrix + j * B->Cols + i) = *(A->Matrix + i * A->Cols + j);
		}
	}
}

void calc_drelu(Matrix_t* vecIn, Matrix_t* vecOut) {
	for (int row = 0; row < vecIn->Rows; row++) {
		*(vecOut->Matrix + row) = *(vecIn->Matrix + row) > 0.0 ? (double)1.0 : (double)0.0;
	}
}

void calc_sigmoid(Matrix_t* vecIn, Matrix_t* vecOut) {
	for (int row = 0; row < vecIn->Rows; row++) {
		*(vecOut->Matrix + row) = 1 / (1 + exp(-1 * (*(vecIn->Matrix + row))));
	}
}

void calc_tanh(Matrix_t* vecIn, Matrix_t* vecOut) {
	for (int row = 0; row < vecIn->Rows; row++) {
		*(vecOut->Matrix + row) = tanh(*(vecIn->Matrix + row));
	}
}

void calc_dtanh(Matrix_t* vecIn, Matrix_t* vecOut) {
	for (int row = 0; row < vecIn->Rows; row++) {
		double a = exp(*(vecIn->Matrix + row));
		double b = exp(-1 * *(vecIn->Matrix + row));
		double num = a - b;
		double den = a + b;
		double val = pow(num, 2) / pow(den, 2);
		*(vecOut->Matrix + row) = 1 - val;
	}
}

void calc_softmax(Matrix_t* vecIn, Matrix_t* vecOut) {
	double sum = 0;
	double temp;
	double max = find_max(vecIn);

	for (int row = 0; row < vecIn->Rows; row++) {
		temp = exp(*(vecIn->Matrix + row) - max);
		*(vecOut->Matrix + row) = temp;
		sum += temp;
	}
	for (int row = 0; row < vecIn->Rows; row++) {
		*(vecOut->Matrix + row) /= sum;
	}
}

void mult(Matrix_t* x, Matrix_t* y, Matrix_t* out) {
	for (int i = 0; i < x->Rows; i++) {
		for (int j = 0; j < x->Cols; j++) {
			*(out->Matrix + i * out->Cols + j) = *(x->Matrix + i * x->Cols + j) * *(y->Matrix + i * y->Cols + j);
		}
	}
}

void add(Matrix_t* x, Matrix_t* y, Matrix_t* out) {
	for (int i = 0; i < x->Rows; i++) {
		for (int j = 0; j < x->Cols; j++) {
			*(out->Matrix + i * out->Cols + j) = *(x->Matrix + i * x->Cols + j) + *(y->Matrix + i * y->Cols + j);
		}
	}
}

void subtract(Matrix_t* x, Matrix_t* y, Matrix_t* out) {
	for (int i = 0; i < x->Rows; i++) {
		for (int j = 0; j < x->Cols; j++) {
			*(out->Matrix + i * out->Cols + j) = *(x->Matrix + i * x->Cols + j) - *(y->Matrix + i * y->Cols + j);
		}
	}
}

void scal_mult(double alpha, Matrix_t* y, Matrix_t* out) {
	for (int i = 0; i < out->Rows; i++) {
		for (int j = 0; j < out->Cols; j++) {
			*(out->Matrix + i * out->Cols + j) = alpha * *(y->Matrix + i * y->Cols + j);
		}
	}
}

void calc_log(Matrix_t* inMat, Matrix_t* outMat) {
	const double SMALL_NUMBER = 10E-10;

	for (int row = 0; row < inMat->Rows; row++) {
		for (int col = 0; col < inMat->Cols; col++) {
			*(outMat->Matrix + row * outMat->Cols + col) = log(*(inMat->Matrix + row * inMat->Cols + col) + SMALL_NUMBER);
		}
	}
}

void zeros(Matrix_t* mat) {
	for (int row = 0; row < mat->Rows; row++) {
		for (int col = 0; col < mat->Cols; col++) {
			*(mat->Matrix + row * mat->Cols + col) = 0;
		}
	}
}

void calc_relu(Matrix_t* vecIn, Matrix_t* vecOut) {
	for (int row = 0; row < vecIn->Rows; row++) {
		*(vecOut->Matrix + row) = *(vecIn->Matrix + row) > 0 ? *(vecIn->Matrix + row) : 0;
	}
}

double sum_vector(Matrix_t* vecIn) {
	double sum = 0.0;

	for (int row = 0; row < vecIn->Rows; row++) {
		sum  += *(vecIn->Matrix + row);
	}

	return sum;
}

double find_max(Matrix_t* vecIn) {
	double max = 0.0;

	for (int row = 0; row < vecIn->Rows; row++) {
		max = *(vecIn->Matrix + row) > max ? *(vecIn->Matrix + row) : max;
	}

	return max;
}

void calc_leaky_relu(int ROWS, float* vecIn, float* vecOut) {
	const float SCALE = (float)0.01;

	for (int row = 0; row < ROWS; row++) {
		*(vecOut + row) = *(vecIn + row) > 0 ? *(vecIn + row) : SCALE * *(vecIn + row);
	}
}

void normalize(Matrix_t* mat) {
	int offset = 0;
#ifdef USE_IMPORT
	double sums[784];
#else
	double sums[4];
#endif
	for (int i = 0; i < sizeof(sums) / sizeof(double); i++) {
		sums[i] = 0;
	}

	for (int v = 0; v < RECORDS; v++) {
		for (int r = 0; r < mat->Rows; r++) {
			sums[r] += mat[v].Matrix[r];
		}
	}

	for (int i = 0; i < mat->Rows; i++) {
		sums[i] /= RECORDS;
	}

	for (int v = 0; v < RECORDS; v++) {
		for (int r = 0; r < mat->Rows; r++) {
			mat[v].Matrix[r] -= sums[r];
		}
	}
}