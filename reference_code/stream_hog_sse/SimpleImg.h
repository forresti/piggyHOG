#ifndef __FORRESTIMG_H__
#define __FORRESTIMG_H__
#include "helpers.h"
//using namespace cv;

#define pixel_t uint8_t //you modify this 
//#define pixel_t float //you modify this 
//#define pixel_t __m128

//TODO: make a templatized version of SimpleImg (put this in a separate test file first)
template<class my_pixel_t>
class SimpleImg2{

};

//no notion of RGB ... just a series of values (pixelType = char, short int, int, etc)
class SimpleImg{
  public:
    SimpleImg(int height, int width, int stride, int n_channels);
    SimpleImg(string fname);
    ~SimpleImg();

    void simple_imwrite(string fname); //write this->data to image file (only tested on jpeg)
    void simple_csvwrite(string fname); //write this->data to csv file

    pixel_t* data;
    int width;
    int stride; //width+padding. note that this is row-major.
    int height;
    int n_channels;
};


#endif

