#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "ForrestImg.h"
#include "helpers.h"
using namespace std;
using namespace cv;

//note: ForrestImg's PIXEL_TYPE is defined in ForrestImg.h
void stream_simple(ForrestImg& img, ForrestImg& outImg){

    assert(img.width == outImg.width);
    assert(img.height == outImg.height);

    for(int y=0; y<img.height; y++){
        for(int x=0; x<img.width; x++){
            outImg.data[y * outImg.stride + x] = img.data[y*img.stride + x];
        }
    }
}

//TODO: init img with some arbitrary values

void fill_img_with_garbage(ForrestImg& img){
    for(int y=0; y<img.height; y++){
        for(int x=0; x<img.width; x++){
            img.data[y*img.stride + x] = (PIXEL_TYPE)(y*img.stride + x); //junk data
        }
    }
}

int main (int argc, char **argv)
{
    int ALIGN_IN_BYTES = 256;

    int height = 640;
    int width = 480*3;
    int stride = width + (ALIGN_IN_BYTES - width%ALIGN_IN_BYTES); //thanks: http://stackoverflow.com/questions/2403631

printf("stide = %d \n", stride);

    ForrestImg img(height, width, stride);
    ForrestImg outImg(height, width, stride);
    fill_img_with_garbage(img);
    int n_iter = 1000;

    double start_timer = read_timer();
    for(int i=0; i<n_iter; i++){
        stream_simple(img, outImg);
        //memcpy(outImg.data, img.data, height * stride * sizeof(PIXEL_TYPE));
    }
    double stream_time = (read_timer() - start_timer) / n_iter;
    double gb_to_copy = width * height * 3 * sizeof(PIXEL_TYPE) / 1e9;
    double gb_per_sec = gb_to_copy / (stream_time/1000); //convert stream_time from ms to sec
    printf("avg stream time = %f ms, %f GB/s \n", stream_time, gb_per_sec);
    

    return 0;
}


