#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cassert>

#include "ForrestImg.h"
#include "helpers.h"
using namespace std;
//using namespace cv;

//note: ForrestImg's pixel_t is defined in ForrestImg.h
void stream_simple(int height, 
		   int width, 
		   int stride,
		   pixel_t *__restrict__ img, 
		   pixel_t *__restrict__ outImg)
{
    for(int y=0; y<height; y++){
        for(int x=0; x<width; x++){
            outImg[y *stride + x] = img[y*stride + x];
        }
    }
}

void fill_img_with_garbage(ForrestImg& img){
    for(int y=0; y<img.height; y++){
        for(int x=0; x<img.width; x++){
            img.data[y*img.stride + x] = (pixel_t)((y*img.stride + x)%256); //junk data
        }
    }
}

int main (int argc, char **argv)
{
    int ALIGN_IN_BYTES = 256;

    int height = 480;
    int width = 640*3;
    int stride = width + (ALIGN_IN_BYTES - width%ALIGN_IN_BYTES); //thanks: http://stackoverflow.com/questions/2403631
    //int stride = width;

printf("stride = %d \n", stride);

    ForrestImg img(height, width, stride);
    ForrestImg outImg(height, width, stride);
    fill_img_with_garbage(img);
    int n_iter = 1000;

    double start_timer = read_timer();
    for(int i=0; i<n_iter; i++){
      stream_simple(img.height, img.width, img.stride, img.data, outImg.data);
        //memcpy(outImg.data, img.data, height * stride * sizeof(pixel_t));
    }
    double stream_time = (read_timer() - start_timer) / n_iter;
    double gb_to_copy = width * height * sizeof(pixel_t) / 1e9;
    double gb_per_sec = gb_to_copy / (stream_time/1000); //convert stream_time from ms to sec
    printf("avg stream time = %f ms, %f GB/s \n", stream_time, gb_per_sec);
    

    return 0;
}


