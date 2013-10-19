#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "helpers.h"
using namespace std;
using namespace cv;
#define eps 0.0001

float ATAN2_TABLE[512][512]; 

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

                float tmp_mag = tmp_gradX*tmp_gradX + tmp_gradY*tmp_gradY;

                if(tmp_mag > max_mag){
                    gradX = tmp_gradX;
                    gradY = tmp_gradY;
                    max_mag = tmp_mag;
                }
            }
            //this is the gradient angle
            //float ori = atan2((double)gradY, (double)gradX); //does float vs. double matter here? 
            //float ori = cv::fastAtan2((double)gradY, (double)gradX);
            float ori = ATAN2_TABLE[(int)gradY + 255][(int)gradX + 255]; //these are already scaled to range of 0-18
            max_mag = sqrt(max_mag); //we've been using magnitude-squared so far

            oriImg.at<float>(y, x) = ori;
            magImg.at<float>(y, x) = max_mag;
        }
    }
}

//simplify to the point of approximating a stream benchmark
inline void grad_stream(Mat img, Mat &gradY_img, Mat &gradX_img){
    for(int y=1; y < (img.rows - 1); y++) //avoid going off the edge of the img
    {
        for(int x=1; x < (img.cols - 1); x++)
        {
#if 0 //basic gradient that looks sorta like stream
            short int gradX;
            short int gradY;
            //float max_mag = 0.0f; //TODO: check range ... can this be uchar?
            int max_mag = 0;

            for(int channel=0; channel<3; channel++){
                //short tmp_gradX = img.at<cv::Vec3b>(y,x+1)[channel] - img.at<cv::Vec3b>(y,x-1)[channel]; 
                short tmp_gradY = img.at<cv::Vec3b>(y+1,x)[channel] - img.at<cv::Vec3b>(y-1,x)[channel];

                //int tmp_mag = tmp_gradX*tmp_gradX + tmp_gradY*tmp_gradY;
                int tmp_mag = tmp_gradY; //stub TODO: remove

                if(tmp_mag > max_mag){
                    //gradX = tmp_gradX;
                    gradY = tmp_gradY;
                    max_mag = tmp_mag;
                }
            }
            //gradX_img.at<short>(y, x) = gradX;
            gradY_img.at<short>(y, x) = gradY;
#endif

#if 1 //trivial stream benchmark
            cv::Vec3b prefetchRGB = img.at<cv::Vec3b>(y,x);
            short tmpResult = 0;
            for(int channel=0; channel<3; channel++){
                //gradY_img.at<short>(y, x) += img.at<cv::Vec3b>(y,x)[channel];
                //gradY_img.at<short>(y, x) += prefetchRGB[channel];
                tmpResult += prefetchRGB[channel];
            }
            gradY_img.at<short>(y, x) = tmpResult;
#endif
        }
    }
}

void writeGradToFile(Mat oriImg, Mat magImg, string detailName){
    oriImg.convertTo(oriImg, CV_8UC1, 255.);
    imwrite((detailName + "_orientations.jpg"), oriImg);

    //magImg.convertTo(magImg, CV_8UC1, 255.);
    imwrite((detailName + "_magnitudes.jpg"), magImg);
}

void init_atan2_table(){
    for (int dy = -255; dy <= 255; ++dy) { //pixels are 0 to 255, so gradient values are -255 to 255
        for (int dx = -255; dx <= 255; ++dx) {
            // Angle in the range [-pi, pi]
            double angle = atan2(static_cast<double>(dy), static_cast<double>(dx));

            // Convert it to the range [9.0, 27.0]
            angle = angle * (9.0 / M_PI) + 18.0;

            // Convert it to the range [0, 18)
            if (angle >= 18.0)
                angle -= 18.0;

            ATAN2_TABLE[dy + 255][dx + 255] = max(angle, 0.0);
        }
    }
}

int main (int argc, char **argv)
{
    init_atan2_table();
    double start_timer;
    Mat img = imread("../../images_640x480/carsgraz_001.image.jpg");
    Mat oriImg(img.rows, img.cols, CV_32FC1); //TODO: replace with aligned mem?
    Mat magImg(img.rows, img.cols, CV_32FC1);

    int n_iter = 10;

#if 0
    start_timer = read_timer();
    for(int i=0; i<10; i++){
        grad_naive(img, oriImg, magImg);    
    }
    double naive_time = (read_timer() - start_timer) / n_iter;
    printf("avg grad_naive time = %f ms \n", naive_time);
    writeGradToFile(oriImg, magImg, "naive");
#endif

    Mat gradX_img(img.rows, img.cols, CV_16SC1); //short int
    Mat gradY_img(img.rows, img.cols, CV_16SC1);
    start_timer = read_timer();
    for(int i=0; i<10; i++){
        grad_stream(img, gradY_img, gradX_img);
    }
    double stream_time = (read_timer() - start_timer) / n_iter;
    printf("avg grad_stream time = %f ms \n", stream_time);
    imwrite("gradX_stream.jpg", gradX_img);
    imwrite("gradY_stream.jpg", gradY_img);


    return 0;
}


