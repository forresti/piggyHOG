#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "helpers_fhog.h"
#include "gradientMex.h"
using namespace std;
using namespace cv;

//input  OpenCV data layout: uchar* [x*d + y*img.cols*d + ch]
//output Matlab data layout: float* [x*img.rows + y + ch*img.rows*img.cols] / 255
float* transpose_opencv_to_matlab(Mat img){
    assert(img.type() == CV_8UC3);
    int h = img.rows; //height
    int w = img.cols; //width
    int d = 3; //nChannels
    float multiplier = 1 / 255.0f; //rescale pixels to range [0 to 1]

    uchar* img_data = &img.data[0];
    float* I = (float*)malloc(h * w * d * sizeof(float)); //img transposed to Matlab data layout. (TODO: pad for SSE?)

    for(int y=0; y<h; y++){
        for(int x=0; x<w; x++){
            for(int ch=0; ch<d; ch++){
                I[x*h + y + ch*w*h] = img_data[x*d + y*w*d + ch] * multiplier; //TODO: cast img_data to float?
            }
        }
    }

    return I;
}

//call Piotr Dollar's FHOG extractor, which was originally designed to have a Matlab front-end
Mat piotr_fhog_wrapper_1img(Mat img){
    int h = img.rows;
    int w = img.cols;
    assert(img.type() == CV_8UC3);
    int d = 3; //nChannels
    bool full = true;
    
    //transpose(img, img); //in-place transpose to col-major to match Piotr's data layout
    //img.convertTo(img, CV_32FC3, 1/255.); //3-channel float, instead of 3-channel uchar
    //assert(img.type() == CV_32FC3);
    //float* I = (float*)&img.data[0]; //note: without the (float*) cast, the compiler complains about converting uchar* to float*. TODO: debug if necessary.
    float* I = transpose_opencv_to_matlab(img);
    float* M = (float*)calloc(h * w, sizeof(float)); //Magnitudes (depth=1)
    float* O = (float*)calloc(h * w, sizeof(float)); //Orientations

  //mGradMag() -> gradMag()
    gradMag(I, M, O, h, w, d, full); //write magnitudes to M and orientations to O

    //defaults from fhog.m
    int binSize = 8; 
    int nOrients = 9;
    int softBin = -1;  
    float clip = 0.2f;
    float* H = (float*)calloc(h * w * (nOrients*3 + 5), sizeof(float)); 

  //mGradHist() -> fhog() 
    fhog(M, O, H, h, w, binSize, nOrients, softBin, clip); //bin and normalize gradients, write HOGs to H
    //void fhog( float *M, float *O, float *H, int h, int w, int binSize,
    //    int nOrients, int softBin, float clip )

    free(I);
    free(O);
    free(M);

    //TODO: return H
    Mat result; //dummy
    return result;
}

void testTranspose(Mat img){
    transpose(img, img);
    //TODO: print out some pointer locations, before and after. 
    //See if the transpose actually moves the data, or if it's just a change in indexing logic.

    //forrestWritePgm(img, "transposed.png"); //TODO: #include common/helpers.h
}

int main (int argc, char **argv)
{
    Mat img = imread("../../images_640x480/carsgraz_001.image.jpg");
    Mat hog = piotr_fhog_wrapper_1img(img); //just for original image scale, for now

    return 0;
}
