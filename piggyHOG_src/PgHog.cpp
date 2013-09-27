#include "PgHog.h"
using namespace std;
using namespace cv;

PgHog::PgHog(){
    //TODO: arctan lookup table
}

PgHog::~PgHog(){

}

//temp debug function declarations
void writeGradToFile(Mat oriImg, Mat gradImg);

//compute the gradient and magnitude at one image location, store the results in oriImg and magImg
void PgHog::gradient(int x, int y, Mat img, Mat &oriImg, Mat &magImg){

    //accesses to 'img' aren't clamped; be careful!
    // make sure x is in range (1, width-1), and y is in range (1, height-1) 

    float gradX;
    float gradY;
    float max_mag = 0.0f; //TODO: check range ... can this be uchar?

    for(int channel=0; channel<3; channel++){
        //TODO: index the data directly instead of using .at
        //float tmp_gradX = img.at<cv::Vec3b>(y,x+1)[channel] - img.at<cv::Vec3b>(y,x-1)[channel];
        //float tmp_gradY = img.at<cv::Vec3b>(y+1,x)[channel] - img.at<cv::Vec3b>(y-1,x)[channel];
        float tmp_gradX = (float)img.at<cv::Vec3b>(y,x-1)[channel] - img.at<cv::Vec3b>(y,x+1)[channel];
        float tmp_gradY = (float)img.at<cv::Vec3b>(y-1,x)[channel] - img.at<cv::Vec3b>(y+1,x)[channel];
        float tmp_mag = tmp_gradX*tmp_gradX + tmp_gradY*tmp_gradY;       

        if(tmp_mag > max_mag){
            gradX = tmp_gradX;
            gradY = tmp_gradY;
            max_mag = tmp_mag;
        } 
    }
    //this is the gradient angle
    float ori = atan2((double)gradY, (double)gradX); //does float vs. double matter here? 

    oriImg.at<float>(y, x) = ori;
    magImg.at<float>(y, x) = max_mag;
}

PgHogContainer PgHog::extract_HOG_oneScale(Mat img, int spatialBinSize){
  //setup
    assert(img.type() == CV_8UC3);
    int sbin = spatialBinSize; //shorthand

    //TODO: possibly go down to 8-bit char
    Mat oriImg(img.rows, img.cols, CV_32FC1); //TODO: replace with aligned mem?
    Mat magImg(img.rows, img.cols, CV_32FC1);

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
                    //update oriImg and magImg at this x,y location
                    PgHog::gradient(hogX*sbin + x, hogY*sbin + y, img, oriImg, magImg);  
                }
            }
        
        }
    }
    writeGradToFile(oriImg, magImg);
}

//----------------- TEMP DEBUG functions below this line ------------------

void writeGradToFile(Mat oriImg, Mat magImg){

    //oriImg.convertTo(oriImg, CV_8UC1, 255.);
    imwrite("PgHog_orientations.jpg", oriImg);
    
    //magImg.convertTo(magImg, CV_8UC1, 255.);
    imwrite("PgHog_magnitudes.jpg", magImg); 
}

