#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cassert>
//#include <xmmintrin.h>
//#include <pmmintrin.h> //for _mm_hadd_pd()

//#include "SimpleImg.h"
#include "SimpleImg.hpp"
//#include "streamHog.h"
//#include "helpers.h"
using namespace std;

SimpleImg* avg_channels(SimpleImg in_img){

    SimpleImg* out_img = new SimpleImg(in_img.height, in_img.width, 1); //1 channel 

    for(int y=0; y<10; y++)
    //for(int y=0; y<in_img.height; y++)
    {
        for(int x=0; x<10; x++)
        //for(int x=0; x<in_img.width; x++)
        {
//            out_img->data[y*(out_img->stride) + x] = (unsigned char)0;
            for(int ch=0; ch<3; ch++){
                out_img->data[y*(out_img->stride) + x] += 10;
                //out_img->data[y*(out_img->stride) + x] += in_img.data[y*in_img.stride + x + ch*in_img.stride*in_img.height] / 3;
//                printf("out_img[y=%d][x=%d] += %d \n", y, x, in_img.data[y*in_img.stride + x + ch*in_img.stride*in_img.height] / 3);
            }
        }
    }
    return out_img;
}

int main(){
    SimpleImg img("../../../images_640x480/carsgraz_001.image.jpg");
    //SimpleImg ori(img.height, img.width, img.stride, 1); //out img has just 1 channel
    //SimpleImg mag(img.height, img.width, img.stride, 1); //out img has just 1 channel

    SimpleImg* outImg = avg_channels(img);

    img.simple_imwrite("./out.jpg");
    //outImg->simple_imwrite("./out.jpg");

    delete outImg;

    return 0;
}
