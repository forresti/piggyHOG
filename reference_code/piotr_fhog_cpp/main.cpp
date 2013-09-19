#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "helpers.h"
#include "gradientMex.h"
using namespace std;
using namespace cv;

//call Piotr Dollar's FHOG extractor, which was originally designed to have a Matlab front-end
Mat piotr_fhog_wrapper_1img(Mat img){

    int h = img.rows;
    int w = img.cols;
    assert(img.type() == CV_8UC3);
    int d = 3; //nChannels
    bool full = true;

  //mGradMag() -> gradMag()
    //void gradMag( float *I, float *M, float *O, int h, int w, int d, bool full )


  //mGradHist() -> fhog()

    //defaults from fhog.m
    int binSize = 8; 
    int nOrients = 9;
    int softBin = -1;  
    float clip = 0.2f;
    
    //void fhog( float *M, float *O, float *H, int h, int w, int binSize,
    //    int nOrients, int softBin, float clip )

}

void testTranspose(Mat img){

    transpose(img, img);
    //TODO: print out some pointer locations, before and after. 
    //See if the transpose actually moves the data, or if it's just a change in indexing logic.

    //forrestWritePgm(img, "transposed.png"); //TODO: #include common/helpers.h
}

int main (int argc, char **argv)
{
//    cv::Mat x;
//    printf("%f \n", foo(1.2345));


    Mat img = imread("../../images_640x480/carsgraz_001.image.jpg");

    Mat hog = piotr_fhog_wrapper_1img(img); //just for original image scale, for now


    return 0;
}
