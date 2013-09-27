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
    PgHogContainer hogResult = pghog.extract_HOG_oneScale(img, spatialBinSize);

    return 0;
}
