#ifndef __TEST_STREAMHOG_H__
#define __TEST_STREAMHOG_H__
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
using namespace std;

bool run_tests_ori_argmax(); //unit test
void test_computeCells_voc5_vs_streamHOG(); //unit test

void test_streamHog_oneScale_default(); //timing experiment
void test_streamHog_pyramid(); //timing experiment

#endif

