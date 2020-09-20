#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "import_data.h"
#include "nnet.h"
#include "wrap_blas.h"

int main() {
	clock_t start, end;
	double cpu_time_used;
	start = clock();

	srand(2);
	//time_t t;
	//srand((unsigned)time(&t));

#ifdef USE_IMPORT
	read_file();
#endif

	Matrix_t x[RECORDS];
	Matrix_t y[RECORDS];
	Matrix_t J;

	//Model_t nnet;
	Matrix_t W[LAYERS];
	Matrix_t b[LAYERS];
	Matrix_t z[LAYERS];
	Matrix_t a[LAYERS];
	Matrix_t dz[LAYERS];
	Matrix_t dW[LAYERS];
	Matrix_t db[LAYERS];

	init_network(W, b, x, y, a, z, dz, dW, db, &J);

	// The weight updates
	for (int epoch = 0; epoch < EPOCHS; epoch++) {
		// Looping over each data point
		for (int sample = 0; sample < RECORDS; sample++) {
			// Get the first data point
			a[0] = x[sample];

			fwd_prop(W, b, a, z);

			calculate_loss(&J, &a[LAYERS - 1], &y[sample]);

			back_prop(W, b, z, a, &y[sample], dW, db, dz);
		}

		update_weights(W, b, dW, db, &J, epoch);
	}

	// Print weights after completion
	for (int layer = 1; layer < LAYERS; layer++) {
		printf("W[%d]\n", layer);
		print(&W[layer]);

		printf("b[%d]\n", layer);
		print(&b[layer]);
	}

	a[0] = x[0];
	fwd_prop(W, b, a, z);
	print(&a[3]);

	a[0] = x[60];
	fwd_prop(W, b, a, z);
	print(&a[3]);

	a[0] = x[120];
	fwd_prop(W, b, a, z);
	print(&a[3]);

	// Delete imported data
	free_data();
	
	for (int sample = 0; sample < RECORDS; sample++) {
		kill_memory(&x[sample]);
		kill_memory(&y[sample]);
	}

	kill_memory(&J);

	for (int layer = 1; layer < LAYERS; layer++) {
		kill_memory(&W[layer]);
		kill_memory(&b[layer]);
		kill_memory(&z[layer]);
		kill_memory(&a[layer]);
		kill_memory(&dz[layer]);
		kill_memory(&dW[layer]);
		kill_memory(&db[layer]);
	}

	end = clock();
	cpu_time_used = (end - start) / (double)CLOCKS_PER_SEC;
	printf("Total execution time %f s\n", cpu_time_used);
	printf("Total forward prop time %f s\n", getFwdTime() / (double)CLOCKS_PER_SEC);
	printf("Total loss time %f s\n", getLossTime() / (double)CLOCKS_PER_SEC);
	printf("Total backward prop time %f s\n", getBkwdTime() / (double)CLOCKS_PER_SEC);

	return 0;
}
