#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "helpers_fhog.h"
#include "common/helpers.h" //CSV file I/O, timers, stuff like that
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

//TODO: test this function
// @input dims:  (y, x, d) = [x*hogHeight + y + d*hogWidth*hogHeight] 
// @output dims: (d, x, y) = [x*hogDepth + y*hogWidth*hogDepth + d] -- same as FFLD.
float* transpose_fhog(float* inHog, int hogHeight, int hogWidth, int hogDepth){
    
    float* outHog = (float*)malloc(hogHeight * hogWidth * hogDepth * sizeof(float)); //TODO: padding for SSE?
    
    //TODO: reorder loops for speed?
    for(int y=0; y<hogHeight; y++){
        for(int x=0; x<hogWidth; x++){
            for(int d=0; d<hogDepth; d++){
                outHog[x*hogDepth + y*hogWidth*hogDepth + d] = inHog[x*hogHeight + y + d*hogWidth*hogHeight];

            }
        }
    }
    return outHog;
}

//this just for one HOG right now ... might tweak it to do whole pyra later
// nRows = 32
// nCols = width*height
void writePyraToCsv(float* hog, int hogHeight, int hogWidth, int hogDepth){
    float* transposedHog = transpose_fhog(hog, hogHeight, hogWidth, hogDepth); 
    //int nlevels = pyramid.levels().size();
    int nlevels = 1;

    for(int level = 0; level < nlevels; level++){
        ostringstream fname;
        fname << "piotr_fhog_results/level"  << level << ".csv"; //TODO: get orig img name into the CSV name.
        int nCols = hogDepth; //one descriptor per row
        int nRows = hogWidth*hogHeight;

        //TODO: also write (depth, width, height) -- in some order -- to the top of the CSV file.
        writeCsv_2dFloat(transposedHog, nRows, nCols, fname.str());
    }
    free(transposedHog);
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
    int hogWidth = w/binSize; //wb in Piotr's code (TODO: play with this)
    int hogHeight = h/binSize; //hb in Piotr's code
    int hogDepth = nOrients*3 + 5;
    float* H = (float*)calloc(hogHeight * hogWidth * hogDepth, sizeof(float)); //TODO: are these dims correct? Should do a "hogH, hogW, hogD?" 

//BEGIN DEBUG
    Mat magnitudes(w, h, CV_32FC1, M); //height and width reversed, because Piotr's data is col major
    transpose(magnitudes, magnitudes);
    magnitudes.convertTo(magnitudes, CV_8UC1, 255.);
    
    imwrite("piotr_magnitudes_cpp.jpg", magnitudes);

//END DEBUG

  //mGradHist() -> fhog()
    //note: fhog internally calculates hogWidth and hogHeight, so we pass the image's height and w into fhog. 
    fhog(M, O, H, h, w, binSize, nOrients, softBin, clip); //bin and normalize gradients, write HOGs to H

    free(I);
    free(O);
    free(M);
    writePyraToCsv(H, hogHeight, hogWidth, hogDepth); //just one HOG, not whole pyra, for now.

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

int main (int argc, char **argv){
    Mat img = imread("../../images_640x480/carsgraz_001.image.jpg");
    Mat hog = piotr_fhog_wrapper_1img(img); //just for original image scale, for now

    return 0;
}

