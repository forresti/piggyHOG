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

int main(){
    SimpleImg img("../../../images_640x480/carsgraz_001.image.jpg");
    SimpleImg ori(img.height, img.width, img.stride, 1); //out img has just 1 channel
    SimpleImg mag(img.height, img.width, img.stride, 1); //out img has just 1 channel

    img.simple_imwrite("./out.jpg");

    return 0;
}
