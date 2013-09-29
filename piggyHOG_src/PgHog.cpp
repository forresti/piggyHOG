#include "PgHog.h"
using namespace std;
using namespace cv;

static inline int clamp(int idx, int min_idx, int max_idx){
    return max(min_idx, min(idx, max_idx));
}

PgHog::PgHog(){

    // Fill the atan2 table (from FFLD) 
    #pragma omp critical
    if (ATAN2_TABLE[0][0] == 0) {
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

}

PgHog::~PgHog(){

}

//temp debug function declarations
void writeGradToFile(Mat oriImg, Mat gradImg);

//compute the gradient and magnitude at one image location, store the results in oriImg and magImg
inline void PgHog::gradient(int x, int y, Mat img, Mat &oriImg, Mat &magImg){
    x = clamp(x, 1, img.cols-1);
    y = clamp(y, 1, img.rows-1);

    float gradX;
    float gradY;
    float max_mag = 0.0f; //TODO: check range ... can this be uchar?

    for(int channel=0; channel<3; channel++){
        float tmp_gradX = img.at<cv::Vec3b>(y,x+1)[channel] - img.at<cv::Vec3b>(y,x-1)[channel];
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
    //float ori = cv::fastAtan2((double)gradY, (double)gradX);
    float ori = ATAN2_TABLE[(int)gradY + 255][(int)gradX + 255]; //these are already scaled to range of 0-18
    max_mag = sqrt(max_mag); //we've been using magnitude-squared so far

    //printf("x = %d, y = %d, gradX = %f, gradY = %f, ori = %f, max_mag = %f \n", x, y, gradX, gradY, ori, max_mag);
    oriImg.at<float>(y, x) = ori;
    magImg.at<float>(y, x) = max_mag;
}

//compute one HOG cell, storing the results in hogResult
inline void PgHog::hogCell(int hogX, int hogY, Mat &oriImg, Mat &magImg, PgHogContainer hogResult){
    //populate this HOG cell by linearly interpolating the oriented gradients 
    //the 'center' of the hog cell is: (hogX+sbin/2, hogY+sbin/2).
    //we do (x,y)=(+/-sbin, +/-sbin) pixels from the center of the hog cell.

    const int sbin=4;
    //int sbin = hogResult.spatialBinSize;
    int hogX_internal = hogX + hogResult.padx; //skip over padding on left side of hogResult.hog
    int hogY_internal = hogY + hogResult.pady; //skip over padding at the top of hogResult.hog

    //TODO: reorganize this math so that it's centered at hogX*sbin+1, and does +/-sbin in all directions?

    int pixelX_start = hogX*sbin - sbin*0.5;
    pixelX_start = clamp(pixelX_start, 0, magImg.cols-1); //not exactly the right logic ... should actually skip the indices that fall off the edge, instead of clamping
    //int pixelX_end = pixelX_start + 2*sbin;    
    
    int pixelY_start = hogY*sbin - sbin*0.5;
    pixelY_start = clamp(pixelY_start, 0, magImg.rows-1);
    //int pixelY_end = pixelY_start + 2*sbin;

    int hogOutputIdx = hogY_internal * hogResult.paddedWidth * hogResult.depth +
                       hogX_internal * hogResult.depth;
    
    //for(int pixelY = pixelY_start; pixelY < pixelY_end; pixelY++){
    //    for(int pixelX = pixelX_start; pixelX < pixelX_end; pixelX++){ 
    for(int offsetY = 0; offsetY < 2*sbin; offsetY++){
        for(int offsetX = 0; offsetX < 2*sbin; offsetX++){

            //location in magImg and gradImg
            int pixelX = pixelX_start + offsetX;
            int pixelY = pixelY_start + offsetY; 
 
            //this pixel's contribution (weight) to our hog cell 
            float weightX = abs(sbin - offsetX) / sbin; //when offset=0, we're at -sbin from hog cell's center. when offset=2*sbin-1, we're +sbin from the center.
            float weightY = abs(sbin - offsetY) / sbin; // TODO: remove division by sbin 

            int oriBin_signed = (int)oriImg.at<float>(pixelY, pixelX); //TODO: just make oriImg a uchar img
            float mag = magImg.at<float>(pixelY, pixelX);

            //hogResult.hog[hogOutputIdx + oriBin_signed] += 1; //test
            //hogResult.hog[hogOutputIdx + oriBin_signed] += mag * 2.0f; //test
            hogResult.hog[hogOutputIdx + oriBin_signed] += mag * weightX * weightY;
        }
    }

    //TODO: oriBin_unsigned ... as a postprocessing step
    //TODO: calculate the sum of this bin and store it (for normalization)
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
    hogResult.padx = 11; //temporary (also, TODO: decide how to handle odd padding sizes)
    hogResult.pady = 6; 
    hogResult.width = round((float)img.cols / (float)spatialBinSize);
    hogResult.height = round((float)img.rows / (float)spatialBinSize);
    hogResult.paddedWidth = hogResult.width + 2*hogResult.padx;
    hogResult.paddedHeight = hogResult.height + 2*hogResult.pady;
    hogResult.depth = 32;
    hogResult.spatialBinSize = spatialBinSize;
    hogResult.hog = (float*)malloc(hogResult.paddedWidth * hogResult.paddedHeight * hogResult.depth * sizeof(float));
    
    //TODO: store normalization results
    //float* norm = malloc(hogResult.paddedWidth * hogResult.paddedHeight * sizeof(float)); 

  //extract features
    for(int hogY = 0; hogY < hogResult.height; hogY++){
        for(int hogX = 0; hogX < hogResult.width; hogX++){

            //calculate gradients 
            for(int y=0; y<spatialBinSize; y++){ //TODO: move these loops into PgHog::gradient()?
                for(int x=0; x<spatialBinSize; x++){
                    //update oriImg and magImg at this x,y location
                    PgHog::gradient(hogX*sbin + x, hogY*sbin + y, img, oriImg, magImg);  
                }
            }

            //HOG cell binning
            if(hogX>0 && hogY>0){
                hogCell(hogX-1, hogY-1, oriImg, magImg, hogResult);
            }

            //TODO: HOG block normalization
        
        }
    }

    //writeGradToFile(oriImg, magImg);
}

//----------------- TEMP DEBUG functions below this line ------------------

void writeGradToFile(Mat oriImg, Mat magImg){
    oriImg.convertTo(oriImg, CV_8UC1, 255.);
    imwrite("PgHog_orientations.jpg", oriImg);
    
    //magImg.convertTo(magImg, CV_8UC1, 255.);
    imwrite("PgHog_magnitudes.jpg", magImg); 
}

