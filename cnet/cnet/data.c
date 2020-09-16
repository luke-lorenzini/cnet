#include "data.h"

#include <stdio.h>


void read_file() {
	double logits[150][4];
	double labels[150][3];

	FILE* fp;
	char buff[255];
	errno_t err;
	char res;
	int recordCount = 0;

	// Open a file
	err = fopen_s(&fp, "iris.data", "r");
	if (err == 0) {
		while (fgets(buff, sizeof(buff) / sizeof(char), fp) != NULL) {
			// Process line here
			printf("%s", buff);

			char localChar;
			int index = 0;

			//while (buff[index] != ',') {
			//	if (buff[index] != '.') {
			//		logits[recordCount][0] =
			//	}
			//}

			logits[recordCount][0] = (int)buff[0] - 48;
			logits[recordCount][1] = (int)buff[4] - 48;
			logits[recordCount][2] = (int)buff[8] - 48;
			logits[recordCount][3] = (int)buff[12] - 48;

			recordCount++;
		}
	}

	if (err == 2) {
		printf("No such file or directory\n");
	}

	printf("\nFound %d records\n", recordCount);

	if (fp) {
		// Close a file
		int res = fclose(fp);
	}
}