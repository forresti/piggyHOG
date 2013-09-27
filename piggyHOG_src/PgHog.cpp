#include "PgHog.h"
using namespace std;
using namespace cv;

PgHog::PgHog(){
    //TODO: arctan lookup table
}

PgHog::~PgHog(){

}

//compute the gradient and magnitude at one image location, store the results in gradImg and magImg
void PgHog::gradient(int x, int y, Mat &gradImg, Mat &magImg){
    

}


PgHogContainer extract_HOG_oneScale(Mat img, int spatialBinSize){
    assert(img.type() == CV_8UC3);

    Mat gradImg(img.rows, img.cols, CV_8UC1);
    Mat magImg(img.rows, img.cols, CV_8UC1);

    PgHogContainer hogResult;
    hogResult.width = round((float)img.cols / (float)spatialBinSize);
    hogResult.height = round((float)img.rows / (float)spatialBinSize);
    hogResult.depth = 32;
    hogResult.hog = (float*)malloc(hogResult.width * hogResult.height * hogResult.depth);

    

}


