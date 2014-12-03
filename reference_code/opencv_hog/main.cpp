
#include <opencv2/opencv.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "helpers.h"
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
    Size outSize = outImg.size();
   
    cv::resize(img,
               outImg,
               outSize,
               0, //scaleX -- default = outSize.width / img.cols
               0, //scaleY -- default = outSize.height / img.rows
               INTER_LINEAR /* use bilinear interpolation */);
              
    return outImg;
}

vector<float> wrappedCvHog(cv::Mat img)
{
    //TODO: pad image to be square -- max(height, width)
    //int dim = max(img.rows, img.cols); 
    img = img.rowRange(0, 480).colRange(0, 480); //trim image to be square -- this fixes the "_M_fill_insert" bug
    int sbin=4;
    bool gamma_corr = true;
    cv::Size win_size(img.rows, img.cols);
    cv::Size block_size(4,4);
    cv::Size block_stride(4,4);
    cv::Size cell_size(sbin,sbin);
    int nOri = 18;

    //off-the-shelf version was from opencv/opencv/samples/ocl/hog.cpp (search for cpu_hog) 
    cv::HOGDescriptor d(win_size, block_size, block_stride, cell_size, nOri, 1, -1,
                              cv::HOGDescriptor::L2Hys, 0.2, gamma_corr, cv::HOGDescriptor::DEFAULT_NLEVELS);

    vector<float> features;
    vector<cv::Point> locations;
    d.compute(img, features, cv::Size(0,0), cv::Size(0,0), locations);
    printf("features.size() = %d \n", (int)features.size());
    return features;
}

//TODO: perhaps return the pyramid?
void wrappedCvHog_pyramid(cv::Mat img)
{
    //TODO: pad image to be square -- max(height, width)
    img = img.rowRange(0, 480).colRange(0, 480); //trim image to be square -- this fixes the "_M_fill_insert" bug
    int sbin=4;
    bool gamma_corr = true;
    cv::Size win_size(img.rows, img.cols);
    cv::Size block_size(4,4);
    cv::Size block_stride(4,4);
    cv::Size cell_size_sbin4(4,4);
    cv::Size cell_size_sbin8(8,8);
    int nOri = 18;

    int nLevels = 30;
    int interval = 10;
    float sc = pow(2, 1 / (float)interval);
    vector< cv::Mat > imgPyramid(nLevels); 
    vector< vector<float> > features(nLevels); //output from HOG code
    vector< vector<cv::Point> > locations(nLevels); //output from HOG code

    cv::HOGDescriptor d_sbin4(win_size, block_size, block_stride, cell_size_sbin4, nOri, 1, -1,
                                  cv::HOGDescriptor::L2Hys, 0.2, gamma_corr, cv::HOGDescriptor::DEFAULT_NLEVELS);

    cv::HOGDescriptor d_sbin8(win_size, block_size, block_stride, cell_size_sbin8, nOri, 1, -1,
                                  cv::HOGDescriptor::L2Hys, 0.2, gamma_corr, cv::HOGDescriptor::DEFAULT_NLEVELS);

    #pragma omp parallel for
    for(int i=0; i<interval; i++){
        float downsampleFactor = 1/pow(sc, i);
        //printf("downsampleFactor = %f \n", downsampleFactor);
        imgPyramid[i] = downsampleWithOpenCV(img, downsampleFactor);
        imgPyramid[i + interval] = downsampleWithOpenCV(img, downsampleFactor/2);
    }

    for(int i=0; i<interval; i++){
        d_sbin4.compute(imgPyramid[i], features[i], cv::Size(0,0), cv::Size(0,0), locations[i]);
        //d_sbin8.compute(imgPyramid[i], features[i + interval], cv::Size(0,0), cv::Size(0,0), locations[i + interval]);
        //d_sbin8.compute(imgPyramid[i + interval], features[i + 2*interval], cv::Size(0,0), cv::Size(0,0), locations[i + 2*interval]);
        printf("features.size() = %d \n", (int)features.size());
    }
    //return features;
}

void wrappedCvHogGPU(cv::Mat img)
{
    img = img.rowRange(0, 480).colRange(0, 480); //trim image to be square -- this fixes the "_M_fill_insert" bug
    int sbin=4;
    bool gamma_corr = true;
    cv::Size win_size(img.rows, img.cols);
    cv::Size block_size(4,4);
    cv::Size block_stride(4,4);
    cv::Size cell_size(sbin,sbin);
    int nOri = 18;

    cv::gpu::HOGDescriptor d(win_size, block_size, block_stride, cell_size, nOri, cv::gpu::HOGDescriptor::DEFAULT_WIN_SIGMA,
                              cv::HOGDescriptor::L2Hys, gamma_corr, cv::HOGDescriptor::DEFAULT_NLEVELS);
    //cv::gpu::HOGDescriptor d; //there are assert statements that pretty much require default configuration
    
    gpu::GpuMat dImg;
    dImg.upload(img); //this hangs, even with a small image.
    gpu::GpuMat features;
    cv::Size win_stride(img.rows, img.cols); //no stride, I guess
    vector<Point> found_locations;

    //d.detect(dImg, found_locations);

    double start = read_timer();
    d.getDescriptors(dImg, win_stride, features);
    double responseTime = read_timer() - start;
    printf("GPU HOG getDescriptors() time = %f \n", responseTime);

    printf("features size = %d \n", features.rows * features.cols);
}

void benchmarkOpenCvHOG()
{
    Mat img = imread("../../images_640x480/carsgraz_001.image.jpg");

//TODO: compile OpenCV w/ GPU support
//TODO: multiscale
#if 1
    double start = read_timer();
    wrappedCvHog(img); 
    double responseTime = read_timer() - start;
    printf("CPU OpenCV HOG time = %f \n", responseTime);
#endif

    //wrappedCvHog_pyramid(img);

#if 0
    start = read_timer();
    wrappedCvHogGPU(img); 
    responseTime = read_timer() - start;
    printf("GPU OpenCV HOG time (including memcpy) = %f \n", responseTime);
#endif
}

int main (int argc, char **argv)
{
    benchmarkOpenCvHOG();    
    return 0;
}
