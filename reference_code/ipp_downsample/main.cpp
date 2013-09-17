#include <opencv2/opencv.hpp>
#include "ipp.h"
#include "ippi.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "common/helpers.h"
#include "helpers_ipp.h"
using namespace std;
using namespace cv;

Mat downsampleWithIPP(Mat img, double scale){  
    int inWidth = img.cols;  
    int inHeight = img.rows;  

    assert(img.type() == CV_8UC3);  
    int nChannels = 3; 
 
    int outWidth = round(inWidth * scale);  
    int outHeight = round(inHeight * scale);  
  
    Mat outImg(outHeight, outWidth, CV_8UC3); //col-major for OpenCV 
    Ipp8u* pSrc = (Ipp8u*)&img.data[0];  
    Ipp8u* pDst = (Ipp8u*)&outImg.data[0];  

    //row-major for IPP
    IppiRect srcRect = {0, 0, inWidth, inHeight};  
    IppiRect dstRect = {0, 0, outWidth, outHeight};  
    IppiSize srcSize = {inWidth, inHeight};  
    IppiSize dstSize = {outWidth, outHeight}; 

    int srcStep = inWidth * nChannels;  
    int dstStep = outWidth * nChannels;  
    IppiPoint dstOffset = {0, 0};  

    int bufsize; 
    CHECK_IPP(ippiResizeGetBufSize(srcRect, dstRect, nChannels, IPPI_INTER_LINEAR, &bufsize));
    Ipp8u* pBuffer = (Ipp8u*)ippMalloc(bufsize);  
 
    int specSize; 
    int initSize;
    CHECK_IPP(ippiResizeGetSize_8u(srcSize, dstSize, ippLinear, 0, &specSize, &initSize)); 
    IppiResizeSpec_32f* pSpec=(IppiResizeSpec_32f*)ippsMalloc_8u(specSize);
    CHECK_IPP(ippiResizeLinearInit_8u(srcSize,
                                      dstSize,  
                                      pSpec));  
 
    //example: https://github.com/albertoruiz/easyVision/blob/master/packages/imagproc/lib/ImagProc/Ipp/auxIpp.c  
    CHECK_IPP(ippiResizeLinear_8u_C3R(pSrc, 
                                      srcStep,  
                                      pDst,  
                                      dstStep,  
                                      dstOffset,  
                                      dstSize,  
                                      ippBorderRepl,
                                      NULL, //borderValue
                                      pSpec, //details of our use case 
                                      pBuffer /* temporary scratch space */ ));  
    ippiFree(pBuffer);  
    return outImg;  
}

void hogPyramid(Mat img){

    int interval = 10;
    float sc = pow(2, 1/interval);

    for(int i=0; i<interval; i++){
        float downsampleFactor = 1/pow(sc, i-1);
        printf("downsampleFactor = %f \n", downsampleFactor);


    }
}


int main (int argc, char **argv){
    Mat img = imread("../../images_640x480/carsgraz_001.image.jpg"); //OpenCV 8U_C3 image

    //one downsample
    double scale = 0.75; //arbitrary
    Mat img_scaled = downsampleWithIPP(img, scale);    
    forrestWritePgm(img_scaled, "carsgraz_001.image_ippScaled.pgm");

    hogPyramid(img);

    return 0;
}
