#ifndef __FORRESTIMG_H__
#define __FORRESTIMG_H__
#include "helpers.h"
//using namespace cv;

#define PIXEL_TYPE int //you modify this 
typedef PIXEL_TYPE pixel_t;

//#define PIXEL_TYPE int //you modify this 


//#define PIXEL_TYPE __m128

//no notion of RGB ... just a series of values (pixelType = char, short int, int, etc)
class ForrestImg{
  public:
    ForrestImg(int height, int width, int stride);
    ~ForrestImg();

    PIXEL_TYPE* data;
    int width;
    int stride; //width+padding. note that this is row-major.
    int height;
};


#endif

