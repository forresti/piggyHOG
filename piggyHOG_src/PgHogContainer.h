#ifndef __PgHogContainer_H__
#define __PgHogContainer_H__
#include <opencv2/opencv.hpp>

//one HOG level
typedef struct PgHogContainer{
    float* hog;
    int width;
    int height;
    int depth; //typically 32
}PgHogContainer;

#endif

