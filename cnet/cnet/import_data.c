#include <stdio.h>
#include <stdlib.h>

#include "import_data.h"

#define BUFF_SIZE 4000

void one_hot(void);

double* logits;
double* labels;
double* oneHotLabels;
int recordCount;

void read_file() {
	logits = (double*)calloc(RECORDS * LOGITS_COLUMNS, sizeof(double));
	labels = (double*)calloc(RECORDS, sizeof(double));
	oneHotLabels = (double*)calloc(RECORDS * LABELS_COLUMNS, sizeof(double));

	FILE* fp;
	char buff[BUFF_SIZE];
	errno_t err;
	//char res;
	recordCount = 0;

	// Open a file
	err = fopen_s(&fp, "mnist_test.csv", "r");
	if (err == 0) {
		while (fgets(buff, BUFF_SIZE, fp) != NULL) {
			// Process line here
			//printf("%s", buff);

			//char localChar;
			int index = 0;

			*(labels + recordCount) = buff[index] - 48;

			for (int i = 2; i < LOGITS_COLUMNS * 2 - 3; i++) {
				if (buff[i] != ',') {
					*(logits + recordCount + i) = buff[i] - 48;
				}
			}

			recordCount++;
		}
	}

	if (err == 2) {
		printf("No such file or directory\n");
	}

	printf("\nFound %d records\n", recordCount);

	//if (fp) {
	//	// Close a file
	//	int res = fclose(fp);
	//}

	one_hot();
}

void one_hot() {
	for (int record = 0; record < recordCount; record++) {
		if (labels[record] == 0) {
			*(oneHotLabels + record * LABELS_COLUMNS + 0) = 1;
		}
		else if (labels[record] == 1) {
			*(oneHotLabels + record * LABELS_COLUMNS + 1) = 1;
		}
		else if (labels[record] == 2) {
			*(oneHotLabels + record * LABELS_COLUMNS + 2) = 1;
		}
		else if (labels[record] == 3) {
			*(oneHotLabels + record * LABELS_COLUMNS + 3) = 1;
		}
		else if (labels[record] == 4) {
			*(oneHotLabels + record * LABELS_COLUMNS + 4) = 1;
		}
		else if (labels[record] == 5) {
			*(oneHotLabels + record * LABELS_COLUMNS + 5) = 1;
		}
		else if (labels[record] == 6) {
			*(oneHotLabels + record * LABELS_COLUMNS + 6) = 1;
		}
		else if (labels[record] == 7) {
			*(oneHotLabels + record * LABELS_COLUMNS + 7) = 1;
		}
		else if (labels[record] == 8) {
			*(oneHotLabels + record * LABELS_COLUMNS + 8) = 1;
		}
		else if (labels[record] == 9) {
			*(oneHotLabels + record * LABELS_COLUMNS + 9) = 1;
		}
	}
}

double get_logit(int index, int pos) {
	return *(logits + index * LOGITS_COLUMNS + pos);
}

double get_label(int index, int pos) {
	return *(oneHotLabels + index * LABELS_COLUMNS + pos);
}

void free_data() {
	free(logits);
	free(labels);
}