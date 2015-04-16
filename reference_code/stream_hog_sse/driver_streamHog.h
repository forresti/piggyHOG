#ifndef __DRIVER_STREAMHOG_H__
#define __DRIVER_STREAMHOG_H__
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
using namespace std;

int get_sbin_for_scale(int scaleIdx, int interval);

void streamHog_pyramid();
void outerLoopParallel_streamHog_pyramid();
#endif

