#ifndef __HELPERS_H__
#define __HELPERS_H__
#include <opencv2/opencv.hpp>
#include <sys/time.h>
#include <stdint.h> //for uintptr_t

#define PIXEL_TYPE uchar 

using namespace std;
using namespace cv;

#define PIXEL_TYPE uchar //you modify this


//macros from voc-release5 fconvsse.cc
#define IS_ALIGNED(ptr) ((((uintptr_t)(ptr)) & 0xF) == 0) 

#if !defined(__APPLE__)
#include <malloc.h>
#define malloc_aligned(a,b) memalign(a,b)
#else
#define malloc_aligned(a,b) malloc(b)
#endif

double read_timer();
std::string forrestGetImgType(int imgTypeInt);

//no notion of RGB ... just a series of values (pixelType = char, short int, int, etc)
template <class pixelType>
class ForrestImg{
  public:
    pixelType img;
    int width;
    int height;
};

#endif

