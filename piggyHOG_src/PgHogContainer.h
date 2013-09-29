#ifndef __PgHogContainer_H__
#define __PgHogContainer_H__
#include <opencv2/opencv.hpp>

//one HOG level
typedef struct PgHogContainer{
    float* hog;
    int width;
    int height;
    int spatialBinSize;
    int padx; //we have (padx/2) empty cells on the left- and right-hand side of the HOG array
    int pady;
    int depth; //typically 32
}PgHogContainer;

#endif

