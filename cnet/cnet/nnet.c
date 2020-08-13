#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "data.h"
#include "nnet.h"

Matrix_t scal_add(double val, Matrix_t* source);
void init_W(Matrix_t* mat, int size);
void trans(Matrix_t* A, Matrix_t* B);
void calc_drelu(Matrix_t* vecIn, Matrix_t* vecOut);
void calc_relu(Matrix_t* vecIn, Matrix_t* vecOut);
void calc_sigmoid(Matrix_t* vecIn, Matrix_t* vecOut);
void calc_softmax(Matrix_t* vecIn, Matrix_t* vecOut);
void mult(Matrix_t* x, Matrix_t* y, Matrix_t* out);
void add(Matrix_t* x, Matrix_t* y, Matrix_t* out);

void calc_log(Matrix_t* inMat, Matrix_t* outMat);

Config_t thing[4] = {
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

// cost
// J += -(y log(a[N]) + (1 - y) log(1 - a[N]))
void calculate_loss(Matrix_t* ptr_J, Matrix_t* ptr_a, Matrix_t* ptr_y) {
	Matrix_t ones;
	Matrix_t temp3;
	Matrix_t temp4;
	Matrix_t temp5;
	Matrix_t A;
	Matrix_t B;

	ones.Rows = ptr_y->Rows;
	ones.Cols = VECTOR_WIDTH;
	ones.Matrix = (double*)calloc(ones.Rows * ones.Cols, sizeof(double));
	for (int i = 0; i < ones.Rows; i++) {
		ones.Matrix[i] = 1;
	}
	temp3.Rows = ptr_y->Rows;
	temp3.Cols = VECTOR_WIDTH;
	temp3.Matrix = (double*)calloc(temp3.Rows * temp3.Cols, sizeof(double));
	temp4.Rows = ptr_y->Rows;
	temp4.Cols = VECTOR_WIDTH;
	temp4.Matrix = (double*)calloc(temp4.Rows * temp4.Cols, sizeof(double));
	temp5.Rows = ptr_y->Rows;
	temp5.Cols = VECTOR_WIDTH;
	temp5.Matrix = (double*)calloc(temp5.Rows * temp5.Cols, sizeof(double));
	A.Rows = ptr_y->Rows;
	A.Cols = VECTOR_WIDTH;
	A.Matrix = (double*)calloc(A.Rows * A.Cols, sizeof(double));
	B.Rows = ptr_y->Rows;
	B.Cols = VECTOR_WIDTH;
	B.Matrix = (double*)calloc(B.Rows * B.Cols, sizeof(double));

	// Calc A
	calc_log(ptr_a, &temp3);
	//printf("ptr_a%d\n", LAYERS - 1);
	//print(&temp3);
	//printf("ptr_y%d\n", SAMPLE_SET - 1);
	//print(&ptr_y[SAMPLE_SET - 1]);
	mult(ptr_y, &temp3, &A);
	//printf("A\n");
	//print(&A);

	// Calc B1
	scopy(ptr_y, &temp4);
	//printf("ptr_y\n");
	//print(&temp4);
	scal(-1, &temp4);
	//printf("ptr_y\n");
	//print(&temp4);
	axpy(1, &ones, &temp4);
	//printf("B1\n");
	//print(&temp4);

	// Calc B2
	scopy(ptr_a, &temp5);
	//printf("ptr_a[N]\n");
	//print(&temp5);
	scal(-1, &temp5);
	//printf("ptr_a[N]\n");
	//print(&temp5);
	axpy(1, &ones, &temp5);
	//printf("ptr_a\n");
	//print(&temp5);
	//printf("ish\n");
	//print(&temp5);
	calc_log(&temp5, &temp5);
	//printf("B2\n");
	//print(&temp5);

	// Calc ptr_J
	mult(&temp4, &temp5, &temp5);
	//printf("ish\n");
	//print(&temp5);
	axpy(1, &A, &temp5);
	//printf("ptr_J\n");
	//print(&temp5);
	axpy(-1, &temp5, ptr_J);
	//printf("ptr_J\n");
	//print(ptr_J);

	kill_memory(&ones);
	kill_memory(&temp3);
	kill_memory(&temp4);
	kill_memory(&temp5);
	kill_memory(&A);
	kill_memory(&B);
}

// fwd prop
// z[n] = (W[n] DOT a[n - 1]) + b[n]
// a[n] = relu(z[n])
// a[N] = sig(z[N])
void fwd_prop(Matrix_t* W, Matrix_t* b, Matrix_t* a, Matrix_t* z) {
	for (int layer = 1; layer < LAYERS; layer++) {
		//printf("W[%d]\n", layer);
		//print(&W[layer]);
		// z = Wx + b
		gemm(&W[layer], &a[layer - 1], &z[layer]);

		axpy(1, &b[layer], &z[layer]);
		//printf("z%d\n", layer);
		//print(&z[layer]);

		// ptr_a = g(z)
		if (layer != (LAYERS - 1)) {
			calc_relu(&z[layer], &a[layer]);
		}
		else {
			//printf("z[%d]\n", layer);
			//print(&z[layer]);
			if (z[layer].Rows == 1) {
				calc_sigmoid(&z[layer], &a[layer]);
			}
			else {
				calc_softmax(&z[layer], &a[layer]);
			}
			//printf("Softmax\n");
			//print(&a[layer]);
			//printf("\n");
		}
		//printf("a[%d]\n", layer);
		//print(&a[layer]);
	}
}

// back prop
// dz[N] = a[N] - y
// dz[n] = WT[n + 1] DOT dz[n + 1] * drelu(z[n])
// db[n] += dz[n]
// dW[n] += dz[n] DOT aT[n - 1]
void back_prop(Matrix_t* W, Matrix_t* b, Matrix_t* z, Matrix_t* a, Matrix_t* y, Matrix_t* dW, Matrix_t* db, Matrix_t* dz) {
	Matrix_t Wt[LAYERS];
	Matrix_t zTemp[LAYERS];
	Matrix_t at[LAYERS];
	Matrix_t temp0[LAYERS];
	Matrix_t temp1[LAYERS];

	for (int idx = 0; idx < LAYERS; idx++) {
		Wt[idx].Rows = thing[idx].Cols;
		Wt[idx].Cols = thing[idx].Rows;
		Wt[idx].Matrix = (double*)calloc(Wt[idx].Rows * Wt[idx].Cols, sizeof(double));

		zTemp[idx].Rows = thing[idx].Rows;
		zTemp[idx].Cols = VECTOR_WIDTH;
		zTemp[idx].Matrix = (double*)calloc(zTemp[idx].Rows * zTemp[idx].Cols, sizeof(double));

		at[idx].Rows = VECTOR_WIDTH;
		at[idx].Cols = thing[idx].Rows;
		at[idx].Matrix = (double*)calloc(at[idx].Rows * at[idx].Cols, sizeof(double));

		temp0[idx].Rows = thing[idx].Rows;
		temp0[idx].Cols = thing[idx].Cols;
		temp0[idx].Matrix = (double*)calloc(temp0[idx].Rows * temp0[idx].Cols, sizeof(double));

		temp1[idx].Rows = thing[idx].Rows;
		temp1[idx].Cols = VECTOR_WIDTH;
		temp1[idx].Matrix = (double*)calloc(temp1[idx].Rows * temp1[idx].Cols, sizeof(double));
	}

	for (int layer = LAYERS - 1; layer > 0; layer--) {
		if (layer == (LAYERS - 1)) {
			// dz[N] // TODO: Fix y, don't use array
			scal(-1, y);
			axpy(1, &a[layer], y);
			scopy(y, &dz[layer]);
		}
		else {
			// dz[n]
			//printf("W%d\n", layer + 1);
			//print(&W[layer + 1]);
			trans(&W[layer + 1], &Wt[layer + 1]);
			//printf("Wt%d\n", layer + 1);
			//print(&Wt[layer + 1]);
			gemm(&Wt[layer + 1], &dz[layer + 1], &temp1[layer]);
			//printf("temp1[%d]\n", layer);
			//print(&temp1[layer]);
			calc_drelu(&z[layer], &zTemp[layer]);
			//printf("zTemp[%d]\n", layer);
			//print(&zTemp[layer]);
			mult(&temp1[layer], &zTemp[layer], &dz[layer]);
		}
		//printf("dz[%d]\n", layer);
		//print(&dz[layer]);

		// db[n]
		axpy(1, &dz[layer], &db[layer]);
		//printf("db[%d]\n", layer);
		//print(&db[layer]);

		// dW[n]
		//printf("a[%d]\n", layer - 1);
		//print(&a[layer - 1]);
		trans(&a[layer - 1], &at[layer - 1]);
		//temp0 wrong dimensions
		gemm(&dz[layer], &at[layer - 1], &temp0[layer]);
		//if (layer == 2) {
		//	printf("temp0[%d]\n", layer);
		//	print(&temp0[layer]);
		//	printf("dw[%d]\n", layer);
		//	print(&dW[layer]);
		//}
		add(&temp0[layer], &dW[layer], &dW[layer]);
		//if (layer == 2) {
		//	printf("dw[%d]\n", layer);
		//	print(&dW[layer]);
		//	printf("\n");
		//}
		//printf("dW[%d]\n", layer);
		//print(&dW[layer]);
		//printf("\n");
	}

	for (int layer = 0; layer < LAYERS; layer++) {
		kill_memory(&Wt[layer]);
		kill_memory(&zTemp[layer]);
		kill_memory(&at[layer]);
		kill_memory(&temp0[layer]);
		kill_memory(&temp1[layer]);
	}
}


void gradcheck(Matrix_t* W, Matrix_t* b, Matrix_t* dW, Matrix_t* db) {
	const int index = 2;
	Matrix_t gradCheck_W;
	Matrix_t gradCheck_Wdiff;
	Matrix_t gradCheck_dW;
	Matrix_t gradCheck_dWdiff;

	gradCheck_W.Rows = W[index].Rows * W[index].Cols + b[index].Rows * b[index].Cols;
	gradCheck_W.Cols = VECTOR_WIDTH;
	gradCheck_W.Matrix = (double*)calloc(gradCheck_W.Rows * gradCheck_W.Cols, sizeof(double));
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
	gradCheck_Wdiff.Matrix = (double*)calloc(gradCheck_Wdiff.Rows * gradCheck_Wdiff.Cols, sizeof(double));

	gradCheck_dW.Rows = dW[index].Rows * dW[index].Cols + db[index].Rows * db[index].Cols;
	gradCheck_dW.Cols = VECTOR_WIDTH;
	gradCheck_dW.Matrix = (double*)calloc(gradCheck_dW.Rows * gradCheck_dW.Cols, sizeof(double));
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
	gradCheck_dWdiff.Matrix = (double*)calloc(gradCheck_dWdiff.Rows * gradCheck_dWdiff.Cols, sizeof(double));
			
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
	//printf("thing\n");
	//print(&gradCheck_dWdiff);
	double dThetaDiff = dnorm(&gradCheck_dWdiff);
	printf("norm(dThetaApprox - dTheta) %f\n", dThetaDiff);

	double grad = dThetaDiff / (dTheta + dThetaApprox);
	printf("grad %f\n", grad);
}

void init_network(Matrix_t* W, Matrix_t* b, Matrix_t* x, Matrix_t* y, Matrix_t* a, Matrix_t* z, Matrix_t* dz, Matrix_t* dW, Matrix_t* db, Matrix_t* J) {
	for (int i = 0; i < SAMPLE_SET; i++) {
		x[i].Rows = ROWS_0;
		x[i].Cols = VECTOR_WIDTH;
		x[i].Matrix = (double*)calloc(x[i].Rows * x[i].Cols, sizeof(double));
		for (int j = 0; j < ROWS_0; j++) {
			x[i].Matrix[j] = IRIS_DATA[i][j];
		}
		//printf("x[%d]\n", i);
		//print(&x[i]);

		y[i].Rows = ROWS_3;
		y[i].Cols = VECTOR_WIDTH;
		y[i].Matrix = (double*)calloc(y[i].Rows * y[i].Cols, sizeof(double));
		for (int j = 0; j < ROWS_3; j++) {
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
		}
		//printf("ptr_y[%d]\n", i);
		//print(&ptr_y[i]);
	}

	J->Rows = VECTOR_WIDTH;
	J->Cols = VECTOR_WIDTH;
	J->Matrix = (double*)calloc(J->Rows * J->Cols, sizeof(double));

	for (int idx = 0; idx < LAYERS; idx++) {
		W[idx].Rows = thing[idx].Rows;
		W[idx].Cols = thing[idx].Cols;
		W[idx].Matrix = (double*)calloc(W[idx].Rows * W[idx].Cols, sizeof(double));
		init_W(&W[idx], W[idx].Cols);

		b[idx].Rows = thing[idx].Rows;
		b[idx].Cols = VECTOR_WIDTH;
		b[idx].Matrix = (double*)calloc(b[idx].Rows * b[idx].Cols, sizeof(double));

		z[idx].Rows = thing[idx].Rows;
		z[idx].Cols = VECTOR_WIDTH;
		z[idx].Matrix = (double*)calloc(z[idx].Rows * z[idx].Cols, sizeof(double));

		a[idx].Rows = thing[idx].Rows;
		a[idx].Cols = VECTOR_WIDTH;
		a[idx].Matrix = (double*)calloc(a[idx].Rows * a[idx].Cols, sizeof(double));

		dz[idx].Rows = thing[idx].Rows;
		dz[idx].Cols = VECTOR_WIDTH;
		dz[idx].Matrix = (double*)calloc(dz[idx].Rows * dz[idx].Cols, sizeof(double));

		dW[idx].Rows = thing[idx].Rows;
		dW[idx].Cols = thing[idx].Cols;
		dW[idx].Matrix = (double*)calloc(dW[idx].Rows * dW[idx].Cols, sizeof(double));

		db[idx].Rows = thing[idx].Rows;
		db[idx].Cols = VECTOR_WIDTH;
		db[idx].Matrix = (double*)calloc(db[idx].Rows * db[idx].Cols, sizeof(double));
	}
}

Matrix_t scal_add(double val, Matrix_t* source) {
	Matrix_t sum;
	sum.Rows = source->Rows;
	sum.Cols = source->Cols;
	sum.Matrix = (double*)calloc(sum.Rows * sum.Cols, sizeof(double));

	for (int i = 0; i < source->Rows; i++) {
		sum.Matrix[i] = *(source->Matrix + i) + val;
	}

	return sum;
}

void init_W(Matrix_t* mat, int size) {
	double SCALE_FACTOR = sqrt(2 / (double)size);
	SCALE_FACTOR = 0.0001;
	double sum = 0;
	int count = 0;

	for (int row = 0; row < mat->Rows; row++) {
		for (int col = 0; col < mat->Cols; col++) {
			*(mat->Matrix + row * mat->Cols + col) = rand() * SCALE_FACTOR;

			sum += *(mat->Matrix + row * mat->Cols + col);
			count++;
		}
	}

	double mu = sum / count;

	for (int row = 0; row < mat->Rows; row++) {
		for (int col = 0; col < mat->Cols; col++) {
			*(mat->Matrix + row * mat->Cols + col) -= mu;
		}
	}
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

void calc_softmax(Matrix_t* vecIn, Matrix_t* vecOut) {
	double sum = 0;
	double temp;
	for (int row = 0; row < vecIn->Rows; row++) {
		temp = exp(*(vecIn->Matrix + row));
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
	const double SMALL_NUMER = 10E-10;

	for (int row = 0; row < inMat->Rows; row++) {
		for (int col = 0; col < inMat->Cols; col++) {
			*(outMat->Matrix + row * outMat->Cols + col) = log(*(inMat->Matrix + row * inMat->Cols + col) + SMALL_NUMER);
		}
	}
}

void kill_memory(Matrix_t* p) {
	free(p->Matrix);
}

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

void zeros(Matrix_t* mat) {
	//for (int row = 0; row < inMat->Rows; row++) {
	//	*(inMat->Matrix + row) = 0;
	//}

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