#ifndef IMPORT_DATA_H
#define IMPORT_DATA_H

//#define USE_DIAGNOSTICS
//#define USE_IMPORT

#ifdef USE_IMPORT
#define RECORDS 100
#define LOGITS_COLUMNS 784
#define LABELS_COLUMNS 10
#else
#define RECORDS 150
#define LOGITS_COLUMNS 4
#define LABELS_COLUMNS 3	// change to 1 or 3
#endif

void read_file(void);
void free_data(void);
double get_logit(int index, int pos);
double get_label(int index, int pos);

#endif

