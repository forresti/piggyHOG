#ifndef __FORRESTIMG_H__
#define __FORRESTIMG_H__
#include "helpers.h"
//using namespace cv;

//#define pixel_t unsigned char //you modify this 
#define pixel_t int16_t
//#define pixel_t float //you modify this 
//#define pixel_t __m128

//no notion of RGB ... just a series of values (pixelType = char, short int, int, etc)
class ForrestImg{
  public:
    ForrestImg(int height, int width, int stride);
    ~ForrestImg();

    pixel_t* data;
    int width;
    int stride; //width+padding. note that this is row-major.
    int height;
};


#endif

