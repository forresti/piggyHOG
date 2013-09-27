#include "PgHog.h"
using namespace std;
using namespace cv;

PgHog::PgHog(){
    //TODO: arctan lookup table
}

PgHog::~PgHog(){

}

//compute the gradient and magnitude at one image location, store the results in gradImg and magImg
void PgHog::gradient(int x, int y, Mat img, Mat &gradImg, Mat &magImg){

    //these aren't clamped; be careful!
    // make sure x is in range (1, width-1), and y is in range (1, height-1) 

    float gradX;
    float gradY;

    for(int channel=0; channel<3; channel++){
        float gradX_tmp = img.at<cv::Vec3b>(y,x+1)[channel] - img.at<cv::Vec3b>(y,x-1)[channel];
    }
}


PgHogContainer PgHog::extract_HOG_oneScale(Mat img, int spatialBinSize){
  //setup
    assert(img.type() == CV_8UC3);

    Mat gradImg(img.rows, img.cols, CV_8UC1); //TODO: replace with aligned mem?
    Mat magImg(img.rows, img.cols, CV_8UC1);

    //hogResult first holds HOG Cells, then is normalized into HOG Blocks.
    PgHogContainer hogResult;
    hogResult.width = round((float)img.cols / (float)spatialBinSize);
    hogResult.height = round((float)img.rows / (float)spatialBinSize);
    hogResult.depth = 32;
    hogResult.hog = (float*)malloc(hogResult.width * hogResult.height * hogResult.depth);

    //TODO: store normalization results
    //float* norm = malloc(hogResult.width * hogResult.height); 

  //extract features
    //for(int hogY = 0; hogY < hogResult.height; hogY++){
    //    for(int hogX = 0; hogX < hogResult.width; hogX++){
    for(int hogY = 1; hogY < hogResult.height-1; hogY++){ //start from 1 to avoid needing to clamp
        for(int hogX = 1; hogX < hogResult.width-1; hogX++){
 
            for(int y=0; y<spatialBinSize; y++){ //TODO: move these loops into PgHog::gradient()?
                for(int x=0; x<spatialBinSize; x++){
                    //update gradImg and magImg at this x,y location
                    PgHog::gradient(hogX + x, hogY + y, img, gradImg, magImg);  
                }
            }

        }
    }
}


