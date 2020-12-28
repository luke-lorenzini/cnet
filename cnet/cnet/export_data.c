#include <stdio.h>

#include "export_data.h"

void export_to_file(Matrix_t* W, Matrix_t* b, int layers) {
	FILE* fp;
	errno_t err;

	err = fopen_s(&fp, "results.csv", "w");

	if (err == 0) {
		for (int l = 1; l < layers; l++) {
			fprintf(fp, "double weight%d[] = {\n", l);
			for (int r = 0; r < W[l].Rows; r++) {
				for (int c = 0; c < W[l].Cols; c++) {
					fprintf(fp, "%f,", W[l].Matrix[(r * W[l].Cols + c)]);
				}
				fprintf(fp, "\n");
			}

			fprintf(fp, "};\n\n");

			fprintf(fp, "double bias%d[] = {\n", l);
			for (int r = 0; r < b[l].Rows; r++) {
				for (int c = 0; c < b[l].Cols; c++) {
					fprintf(fp, "%f,", b[l].Matrix[r]);
				}
				fprintf(fp, "\n");
			}

			fprintf(fp, "};\n\n");
		}
	}

	if (fp) {
		// Close a file
		int res = fclose(fp);
	}
}