
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
double runIppConvolution(int imgRows, int imgCols, int kernelRows, int kernelCols, string dataType)
{
    Mat img;
    if(dataType == "8u")
        img = Mat::zeros(imgRows, imgCols, CV_8UC1);
    if(dataType == "32f")
        img = Mat::zeros(imgRows, imgCols, CV_32FC1);

    Mat outImg = img.clone(); //allocate space for the convolution results 
    
    int step = imgCols; //pitch
    const Ipp32s kernel_int[kernelRows*kernelCols]; //zeros
    const Ipp32f kernel_float[kernelRows*kernelCols];
    IppiSize kernelSize = {kernelCols, kernelRows};
    IppiSize dstRoiSize = {imgCols - kernelSize.width + 1, imgRows - kernelSize.height + 1};
    IppiPoint anchor = {2,2};
    int divisor = 1;
    IppStatus status;

    double start = read_timer();
    if(dataType == "8u"){
        status = ippiFilter_8u_C1R((const Ipp8u*)&img.data[0], step, 
                                   (Ipp8u*)&outImg.data[0], step, dstRoiSize, 
                                   kernel_int, kernelSize, anchor, divisor);
    }
    if(dataType == "32f"){
        step*=4; //pitch is in bytes
        status = ippiFilter_32f_C1R((const Ipp32f*)&img.data[0], step,
                                   (Ipp32f*)&outImg.data[0], step, dstRoiSize,          
                                   kernel_float, kernelSize, anchor);
    }
    double execTime = read_timer() - start;
    if(status != 0) 
        cout << "IppiFilter error status " << ippGetStatusString(status) << endl;
    return execTime;
}

void benchmarkIppConvolution()
{
    int imgRows = 9000;
    int imgCols = 9000;
    string dataType = "8u";
    //for(int kernelSize=2; kernelSize<128; kernelSize*=2)
    int kernelSize=5;
    for(int i=0; i<10; i++)
    {
        double execTime = runIppConvolution(imgRows, imgCols, kernelSize, kernelSize, dataType);
        printf("imgSize = %dx%d,  kernelSize = %d,  execTime = %f ms \n", imgCols, imgRows, kernelSize, execTime);
    }

    kernelSize = 4;
    for(int imgSize=256; imgSize<5000; imgSize*=2) //imgSize = imgRows = imgCols
    {
        double execTime = runIppConvolution(imgSize, imgSize, kernelSize, kernelSize, dataType);
        printf("imgSize = %dx%d,  kernelSize = %d,  execTime = %f ms \n", imgSize, imgSize, kernelSize, execTime);
    }

}

int main (int argc, char **argv)
{
    //double execTime = runIppConvolution(200, 200, 5, 5);
    //printf("execTime = %f \n", execTime);
    benchmarkIppConvolution();

    return 0;
}
