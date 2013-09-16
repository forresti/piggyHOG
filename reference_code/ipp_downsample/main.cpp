
#include <opencv2/opencv.hpp>
#include "ipp.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "common/helpers.h"
#include "helpers_ipp.h"
using namespace std;
using namespace cv;


//from my StackOverflow post: http://stackoverflow.com/questions/13465914/using-opencv-mat-images-with-intel-ipp
void demoIppConvolution(){
    Mat img = imread("./Lena.pgm"); //OpenCV 8U_C1 image
    Mat outImg = img.clone(); //allocate space for convolution results

    int step = img.cols*3; //pitch
    const Ipp32s kernel[9] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
    IppiSize kernelSize = {3,3};
    IppiSize dstRoiSize = {img.cols - kernelSize.width + 1, img.rows - kernelSize.height + 1};
    IppiPoint anchor = {2,2};
    int divisor = 1;

    IppStatus status = ippiFilter_8u_C3R((const Ipp8u*)&img.data[0], step,
                                         (Ipp8u*)&outImg.data[0], step, dstRoiSize,
                                         kernel, kernelSize, anchor, divisor);
    forrestWritePgm(outImg, "Lena_ipp.pgm");  
}

int main (int argc, char **argv){
    demoIppConvolution();

    return 0;
}
