
//#include <cuda.h>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "hog.hpp" //https://github.com/lastlegion/hog
#include "helpers.h"
using namespace std;

int main (int argc, char **argv)
{
    cv::Mat x;
    printf("%f \n", foo(1.2345));
// What will this print
float f = 1.0;
int  *p = (int*)(&f);

printf("%d\n", *p);

    return 0;
}
