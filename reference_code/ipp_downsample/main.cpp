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

vector<Mat> downsamplePyramid(Mat img){
    int interval = 10;
    float sc = pow(2, 1 / (float)interval);
    vector<Mat> imgPyramid(interval*2); //100% down to 25% of orig size (two octaves, 10 scales per octave)

    for(int i=0; i<interval; i++){
        float downsampleFactor = 1/pow(sc, i);
        printf("downsampleFactor = %f \n", downsampleFactor);
        imgPyramid[i] = downsampleWithIPP(img, downsampleFactor); //TODO: catch return images, put them in a vector<Mat>
        //imgPyramid[i+interval] = downsampleWithIPP(img, downsampleFactor/2);
        imgPyramid[i+interval] = downsampleWithIPP(imgPyramid[i], downsampleFactor); //start from already downsampled img, go down an other octave
    }
    return imgPyramid;
}

void downsampleDemo(Mat img){
    //one downsample
    double scale = 0.75; //arbitrary
    Mat img_scaled = downsampleWithIPP(img, scale);    
    forrestWritePgm(img_scaled, "carsgraz_001.image_ippScaled.pgm");
}

int main (int argc, char **argv){
    Mat img = imread("../../images_640x480/carsgraz_001.image.jpg"); //OpenCV 8U_C3 image

    //one downsample
    downsampleDemo(img);

    double start_pyra = read_timer();
    vector<Mat> imgPyramid = downsamplePyramid(img);
    double time_pyra = read_timer() - start_pyra;
    printf("    downsample image for HOG pyramid in %f ms \n", time_pyra);

    return 0;
}
