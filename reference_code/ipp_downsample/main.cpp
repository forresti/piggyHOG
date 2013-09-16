
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

//returns execution time of ippiFilter
//double runIppConvolution(int imgRows, int imgCols, int kernelRows, int kernelCols, string dataType)
#if 0
double demoIppConvolution(int kernelRows, int kernelCols) //just use Lena.png
{
    Mat img = imread("./Lena.pgm");
#if 0
    Mat img;
    if(dataType == "8u")
        img = Mat::zeros(imgRows, imgCols, CV_8UC1);
    if(dataType == "32f")
        img = Mat::zeros(imgRows, imgCols, CV_32FC1);
#endif
    Mat outImg = img.clone(); //allocate space for the convolution results 

    int imgCols = img.cols;
    int imgRows = img.rows;

    int step = imgCols*3; //pitch
    const Ipp32s kernel_int[kernelRows*kernelCols]; //zeros
    const Ipp32f kernel_float[kernelRows*kernelCols];
    IppiSize kernelSize = {kernelCols, kernelRows};
    IppiSize dstRoiSize = {imgCols - kernelSize.width + 1, imgRows - kernelSize.height + 1};
    IppiPoint anchor = {2,2};
    int divisor = 1;
    IppStatus status;

    double start = read_timer();
    //if(dataType == "8u"){
        status = ippiFilter_8u_C3R((const Ipp8u*)&img.data[0], step, 
                                   (Ipp8u*)&outImg.data[0], step, dstRoiSize, 
                                   kernel_int, kernelSize, anchor, divisor);
    //}
#if 0
    if(dataType == "32f"){
        step*=4; //pitch is in bytes
        status = ippiFilter_32f_C1R((const Ipp32f*)&img.data[0], step,
                                   (Ipp32f*)&outImg.data[0], step, dstRoiSize,          
                                   kernel_float, kernelSize, anchor);
    }
#endif
    double execTime = read_timer() - start;
    if(status != 0) 
        cout << "IppiFilter error status " << ippGetStatusString(status) << endl;

    forrestWritePgm(outImg, "Lena_ipp.pgm");    

    return execTime;
}

void benchmarkIppConvolution()
{
    int imgRows = 9000;
    int imgCols = 9000;
    string dataType = "32f";
    //for(int kernelSize=2; kernelSize<128; kernelSize*=2)
    int kernelSize=5;
    for(int i=0; i<10; i++)
    {
        //double execTime = runIppConvolution(imgRows, imgCols, kernelSize, kernelSize, dataType);
        double execTime = demoIppConvolution(kernelSize, kernelSize);
        printf("imgSize = %dx%d,  kernelSize = %d,  execTime = %f ms \n", imgCols, imgRows, kernelSize, execTime);
    }
#if 0
    kernelSize = 4;
    for(int imgSize=256; imgSize<5000; imgSize*=2) //imgSize = imgRows = imgCols
    {
        double execTime = runIppConvolution(imgSize, imgSize, kernelSize, kernelSize, dataType);
        printf("imgSize = %dx%d,  kernelSize = %d,  execTime = %f ms \n", imgSize, imgSize, kernelSize, execTime);
    }
#endif
}
#endif

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

int main (int argc, char **argv)
{
    //double execTime = runIppConvolution(200, 200, 5, 5);
    //printf("execTime = %f \n", execTime);
    //benchmarkIppConvolution();
    demoIppConvolution();

    return 0;
}
