#include <opencv2/opencv.hpp>
#include "ipp.h"
#include "ippi.h"
#include "omp.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "common/helpers.h"
#include "helpers_ipp.h"
using namespace std;
using namespace cv;

//use OpenCV's bilinear filter downsampling
Mat downsampleWithOpenCV(Mat img, double scale){
    int inWidth = img.cols;
    int inHeight = img.rows;
    assert(img.type() == CV_8UC3);
    int nChannels = 3;

    int outWidth = round(inWidth * scale);
    int outHeight = round(inHeight * scale);
    Mat outImg(outHeight, outWidth, CV_8UC3); //col-major for OpenCV 

    //outImg 'knows' the downsampled dimensions
    cv::resize(img, outImg); //using default bilinear interpolation (INTER_LINEAR) 
    return outImg;
}

//use Intel IPP's bilinear filter downsampling
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

    omp_set_num_threads(5); //TODO: be careful with num threads. on R8, 2-6 threads is good, more is just noisy

    #pragma omp parallel for
    for(int i=0; i<interval; i++){
        //printf("omp_get_num_threads = %d \n", omp_get_num_threads());

        float downsampleFactor = 1/pow(sc, i);
        //printf("downsampleFactor = %f \n", downsampleFactor);

#if 1 //IPP downsample
        imgPyramid[i] = downsampleWithIPP(img, downsampleFactor); 
        imgPyramid[i+interval] = downsampleWithIPP(img, downsampleFactor/2);
        //imgPyramid[i+interval] = downsampleWithIPP(imgPyramid[i], downsampleFactor); //start from already downsampled img, go down an other octave
#endif
    }
    return imgPyramid;
}

//TODO: delete downsampleDemo()
void downsampleDemo(Mat img){
    //one downsample
    double scale = 0.75; //arbitrary

#if 1 //OpenCV downsample
    Mat img_scaled = downsampleWithOpenCV(img, scale);
    forrestWritePgm(img_scaled, "carsgraz_001.image_opencvScaled.pgm");
#endif

#if 0 //IPP downsample
    Mat img_scaled = downsampleWithIPP(img, scale);    
    forrestWritePgm(img_scaled, "carsgraz_001.image_ippScaled.pgm");
#endif
}

int main (int argc, char **argv){
    Mat img = imread("../../images_640x480/carsgraz_001.image.jpg"); //OpenCV 8U_C3 image

    //one downsample
    downsampleDemo(img);

    //downsample pyramid
    double start_pyra = read_timer();
    vector<Mat> imgPyramid = downsamplePyramid(img);
    double time_pyra = read_timer() - start_pyra;
    printf("    downsample image for HOG pyramid in %f ms \n", time_pyra);
    //TODO: have a function to write imgPyramid out to img files

    return 0;
}
