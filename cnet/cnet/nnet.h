#ifndef NNET_H
#define NNET_H

#include "wrap_blas.h"

void scal_mult(double alpha, Matrix_t* y, Matrix_t* out);
void print(Matrix_t*);
void kill_memory(Matrix_t* p);
void zeros(Matrix_t* mat);
void subtract(Matrix_t* x, Matrix_t* y, Matrix_t* out);

// W[l] = (n[l], n[l-1])
// b[l] = (n[l], 1)
// dW[l] = (n[l], n[l-1])
// db[l] = (n[l], 1)

#define LAYERS 4
#define SAMPLE_SET	150

#define VECTOR_WIDTH  1

#define AAAA 4
#define BBBB 16
#define CCCC 16
#define DDDD 1

// Layers 20, 30, 50, 1
// Layer 0 (3, 1)
#define ROWS_0 AAAA
#define COLS_0 VECTOR_WIDTH

// Layer 1 (30, 20)
#define ROWS_1 BBBB
#define COLS_1 AAAA

// Layer 2 (50, 30)
#define ROWS_2 CCCC
#define COLS_2 BBBB

// Layer 3 (1, 50)
#define ROWS_3 DDDD
#define COLS_3 CCCC

typedef struct {
	int Rows;
	int Cols;
} Config_t;

void calculate_loss(Matrix_t* J, Matrix_t* a, Matrix_t* y);
void fwd_prop(Matrix_t* W, Matrix_t* b, Matrix_t* a, Matrix_t* z);
void back_prop(Matrix_t* W, Matrix_t* b, Matrix_t* z, Matrix_t* a, Matrix_t* y, Matrix_t* dW, Matrix_t* db, Matrix_t* dz, int sample);
void init_network(Matrix_t* x, Matrix_t* y, Matrix_t* W, Matrix_t* b, Matrix_t* a, Matrix_t* z, Matrix_t* dz, Matrix_t* dW, Matrix_t* db, Matrix_t* J);
void gradcheck(Matrix_t* W, Matrix_t* b, Matrix_t* dW, Matrix_t* db);

#endif