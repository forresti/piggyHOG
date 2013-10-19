#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "helpers.h"
using namespace std;
using namespace cv;
#define eps 0.0001

static inline int clamp(int idx, int min_idx, int max_idx){
    return max(min_idx, min(idx, max_idx));
}

//roughly the grad impl that I've been using in piggyHOG
inline void grad_naive(Mat img, Mat &oriImg, Mat &magImg){

    //TODO: index x and y from 0, and clamp?

    for(int y=1; y < (img.rows - 1); y++) //avoid going off the edge of the img
    {
        for(int x=1; x < (img.cols - 1); x++)
        {
            float gradX;
            float gradY;
            float max_mag = 0.0f; //TODO: check range ... can this be uchar?

            for(int channel=0; channel<3; channel++){
                float tmp_gradX = img.at<cv::Vec3b>(y,x+1)[channel] - img.at<cv::Vec3b>(y,x-1)[channel]; //TODO: cast pixels to float (to avoid overflow/underflow?)
                float tmp_gradY = img.at<cv::Vec3b>(y+1,x)[channel] - img.at<cv::Vec3b>(y-1,x)[channel];

                //indexing the data directly instead of using .at -- doesn't help with perf.
                //float tmp_gradX = img.data[y*img.rows*3 + (x+1)*3 + channel] - img.data[y*img.rows*3 + (x-1)*3 + channel];
                //float tmp_gradY = img.data[(y+1)*img.rows*3 + x*3 + channel] - img.data[(y-1)*img.rows*3 + x*3 + channel];
                float tmp_mag = tmp_gradX*tmp_gradX + tmp_gradY*tmp_gradY;

                if(tmp_mag > max_mag){
                    gradX = tmp_gradX;
                    gradY = tmp_gradY;
                    max_mag = tmp_mag;
                }
            }
            //this is the gradient angle
            //float ori = atan2((double)gradY, (double)gradX); //does float vs. double matter here? 
            float ori = cv::fastAtan2((double)gradY, (double)gradX);
            //float ori = ATAN2_TABLE[(int)gradY + 255][(int)gradX + 255]; //these are already scaled to range of 0-18
            max_mag = sqrt(max_mag); //we've been using magnitude-squared so far

            oriImg.at<float>(y, x) = ori;
            magImg.at<float>(y, x) = max_mag;
        }
    }
}

void writeGradToFile(Mat oriImg, Mat magImg, string detailName){
    oriImg.convertTo(oriImg, CV_8UC1, 255.);
    imwrite("PgHog_orientations.jpg", oriImg);

    //magImg.convertTo(magImg, CV_8UC1, 255.);
    imwrite("PgHog_magnitudes.jpg", magImg);
}

int main (int argc, char **argv)
{
    Mat img = imread("../../images_640x480/carsgraz_001.image.jpg");
    Mat oriImg(img.rows, img.cols, CV_32FC1); //TODO: replace with aligned mem?
    Mat magImg(img.rows, img.cols, CV_32FC1);

    int n_iter = 10;

    double start_timer = read_timer();
    for(int i=0; i<10; i++){
        grad_naive(img, oriImg, magImg);    
    }
    double naive_time = (read_timer() - start_timer) / n_iter;
    printf("avg grad_naive time = %f ms \n", naive_time);

    writeGradToFile(oriImg, magImg, "naive");

    return 0;
}




