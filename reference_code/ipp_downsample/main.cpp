
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
    Mat img = imread("./Lena.pgm"); //OpenCV 8U_C3 image
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

void downsampleWithIPP(Mat img, scale){  
i    int inWidth = img.cols;  
    int inHeight = img.rows;  
    int nChannels = img.depth;  
  
    assert(nChannels == 3);  
    assert(img.type() == CV_8UC3);  
  
    int outWidth = round(inWidth * scale);  
    int outHeight = round(inHeight * scale);  
  
    Mat outImg(outWidth, outHeight, CV_8UC3);  
    Ipp8u* pSrc = (Ipp8u*)&img.data[0];  
    Ipp8u* pDst = (Ipp8u*)&outImg.data[0];  
  
    IppiRect srcRect = {0, 0, inWidth, inHeight};  
    IppiRect dstRect = {0, 0, outWidth, outHeight};  
    IppiSize srcSize = {inWidth, inHeight};  
    IppiSize dstSize = {outWidth, outHeight};  
  
    int srcStep = inWidth * nChannels;  
    int dstStep = outWidth * nChannels;  
    IppiPoint dstOffset = {0, 0};  
  
    IppiStatus status = ippiResizeGetBufSize(srcRect, dstRect, nChannels, IPPI_INTER_LINEAR, &bufsize); 
    if(status != ippStsNoEr)  
        return -1;  
  
    Ipp8u* pBuffer = (Ipp8u*)ippMalloc(bufsize);  
    if(pBuffer == NULL)  
        return -1;  
  
    IppiResizeSpec_32f pspec; //TODO: ippiMalloc this if we have problems  
  
    IppStatus ippiResizeLinearInit_8u_C3R(IppiSize srcSize,  
                                          IppiSize dstSize,  
                                          &pSpec);  
  
    //http://software.intel.com/sites/products/documentation/doclib/ipp_sa/71/ipp_manual/IPPI/ippi_ch12/functn_ResizeLinear.htm 
    //example: https://github.com/albertoruiz/easyVision/blob/master/packages/imagproc/lib/ImagProc/Ipp/auxIpp.c  
    IppStatus ippiResizeLinear_8u_C3R(const Ipp8u* pSrc,  
                                      Ipp32s srcStep,  
                                      Ipp8u* pDst,  
                                      Ipp32s dstStep,  
                                      IppiPoint dstOffset,  
                                      IppiSize dstSize,  
                                      IppiBorderType border, //TODO ... let's say ippBorderConst or ippBorderRepl  
                                      Ippi8u* borderValue, //NULL -- as in https://github.com/albertoruiz/easyVision/blob/master/packages/imagproc/lib/ImagProc/Ipp/auxIpp.c  
                                      IppiResizeSpec_32f* pSpec, //might need to do '&pSpec'  
                                      Ipp8u* pBuffer /* temporary scratch space */ );  
  
    ippiFree(pBuffer);  
  
    return outImg;  
}


int main (int argc, char **argv){
    demoIppConvolution();

    Mat img = imread("./Lena.pgm"); //OpenCV 8U_C3 image
    int scale = 0.75; //arbitrary

    Mat img_scaled = downsampleWithIPP(img, scale);    

    return 0;
}
