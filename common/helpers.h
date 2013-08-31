#ifndef __HELPERS_H__
#define __HELPERS_H__
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <sys/time.h>
using namespace std;

void writeCsv_2dFloat(vector<float> vec, int nRows, int nCols, string fname);
double read_timer();

#endif

