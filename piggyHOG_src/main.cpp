#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "common/helpers.h"
#include "PgHog.h"
using namespace std;

int main (int argc, char **argv)
{
    Mat img = imread("../images_640x480/carsgraz_001.image.jpg"); //OpenCV 8U_C3 image

    PgHog pghog;
    int spatialBinSize = 4;


    double start_timer = read_timer();
    PgHogContainer hogResult = pghog.extract_HOG_oneScale(img, spatialBinSize);
    double time_one_scale = read_timer() - start_timer;

    printf("time for [whatever's implemented so far] for 1 scale: %f ms \n", time_one_scale);

    return 0;
}
