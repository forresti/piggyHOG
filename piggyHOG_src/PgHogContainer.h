#ifndef __PgHogContainer_H__
#define __PgHogContainer_H__
#include <opencv2/opencv.hpp>

//one HOG level
typedef struct PgHogContainer{
    float* hog;
    int width;
    int height;
    int spatialBinSize;
    int padx; //we have padx empty cells on the left- and right-hand side of the HOG array (e.g. padx=11, sbin=4, then we have 44 empty HOG descriptors on the left side, and on the right side)
    int pady;
    int depth; //typically 32
}PgHogContainer;

#endif

