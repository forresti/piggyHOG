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

void avg_channels(SimpleImg &in_img, SimpleImg &out_img)
{
    for(int y=0; y<in_img.height; y++)
    {
        for(int x=0; x<in_img.width; x++)
        {
            out_img.data[y*(out_img.stride) + x] = (unsigned char)0;
            for(int ch=0; ch<3; ch++){
                //out_img.data[y*(out_img.stride) + x] += 10;
                out_img.data[y*(out_img.stride) + x] += in_img.data[y*in_img.stride + x + ch*in_img.stride*in_img.height] / 3;
            }
        }
    }
}

void SimpleImg_test(){
    SimpleImg img("../../../images_640x480/carsgraz_001.image.jpg");

    SimpleImg out_img(img.height, img.width, 1);
    avg_channels(img, out_img); //out_img gets filled in

    out_img.simple_imwrite("./out.jpg");
}

int main (int argc, char **argv)
{
    SimpleImg_test();
    
    return 0;

}


