#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "helpers.h"
using namespace std;
using namespace cv;

//note: ForrestImg's PIXEL_TYPE is defined in helpers.h
void stream_simple(ForrestImg& img, ForrestImg& outImg){

    assert(img.width == outImg.width);
    assert(img.height == outImg.height);

    for(int y=0; y<img.width; y++){
        for(int x=0; x<width; x++){
            outImg.data[y * outImg.stride + x] = img.data[y*img.stride + x]
        }
    }
}

//TODO: init img with some arbitrary values

int main (int argc, char **argv)
{
    ForrestImg img;
    ForrestImg outImg;

    int n_iter = 10;

    double start_timer = read_timer();
    double stream_time = (read_timer() - start_timer) / n_iter;
    printf("avg stream time = %f ms \n", stream_time);


    return 0;
}


