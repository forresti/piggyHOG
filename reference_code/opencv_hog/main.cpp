
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

vector<float> wrappedCvHog(cv::Mat img)
{
    bool gamma_corr = true;
    cv::Size win_size(img.rows, img.cols);
    //int c = 4;
    //cv::Size block_size(c,c);
    //cv::Size block_stride(c,c); //I think this is to select whether to do overlapping blocks
    //cv::Size cell_size(c,c);
    cv::Size block_size(16,16);
    cv::Size block_stride(8,8);
    cv::Size cell_size(8,8);
    int nOri = 15;

    //off-the-shelf version was from opencv/opencv/samples/ocl/hog.cpp (search for cpu_hog) 
    cv::HOGDescriptor d(win_size, block_size, block_stride, cell_size, nOri, 1, -1,
                              cv::HOGDescriptor::L2Hys, 0.2, gamma_corr, cv::HOGDescriptor::DEFAULT_NLEVELS);

    vector<float> features;
    vector<cv::Point> locations;
    d.compute(img, features, cv::Size(0,0), cv::Size(0,0), locations);
    printf("features.size() = %d \n", (int)features.size());
    return features;
}

void wrappedCvHogGPU(cv::Mat img)
{
    bool gamma_corr = true;
    cv::Size win_size(img.rows, img.cols);
    //int c = 4;
    //cv::Size block_size(c,c);
    //cv::Size block_stride(c,c); //I think this is to select whether to do overlapping blocks
    //cv::Size cell_size(c,c);
    cv::Size block_size(16,16);
    cv::Size block_stride(8,8);
    cv::Size cell_size(8,8);
    int nOri = 15;

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
    //Mat img = imread("../forrest_hacked_OpenCV_2_Cookbook_Code/9k_x_9k.png");
    //Mat img = imread("../forrest_hacked_OpenCV_2_Cookbook_Code/Lena_orig.png");
    cv::cvtColor(img, img, CV_RGB2GRAY);   

//TODO: fix errors
//TODO: compile OpenCV w/ GPU support
//TODO: multiscale
    double start = read_timer();
    wrappedCvHog(img); 
    double responseTime = read_timer() - start;
    printf("CPU OpenCV HOG time = %f \n", responseTime);

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
