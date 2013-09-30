#include "PgHog.h"
#include <iomanip>
#include <iostream>
#include <fstream>
using namespace std;
using namespace cv;

#define eps 0.0001

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
void writeHogCellsToFile(PgHogContainer hogResult);

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
// only compute the contrast-sensitive features (0 to 360 degrees)
inline void PgHog::hogCell(int hogX, int hogY, Mat &oriImg, Mat &magImg, PgHogContainer hogResult){
    //populate this HOG cell by linearly interpolating the oriented gradients 
    //the 'center' of the hog cell is: (hogX+sbin/2, hogY+sbin/2).
    //we do (x,y)=(+/-sbin, +/-sbin) pixels from the center of the hog cell.

    const int sbin=4;
    //int sbin = hogResult.spatialBinSize;
    int hogX_internal = hogX + hogResult.padx; //skip over padding on left side of hogResult.hog
    int hogY_internal = hogY + hogResult.pady; //skip over padding at the top of hogResult.hog

    int pixelX_start = hogX*sbin - sbin*0.5f;
    pixelX_start = clamp(pixelX_start, 0, magImg.cols-1); //not exactly the right logic ... should actually skip the indices that fall off the edge, instead of clamping
    //int pixelX_end = pixelX_start + 2*sbin;    
    
    int pixelY_start = hogY*sbin - sbin*0.5f;
    pixelY_start = clamp(pixelY_start, 0, magImg.rows-1);
    //int pixelY_end = pixelY_start + 2*sbin;

    int hogOutputIdx = hogY_internal * hogResult.paddedWidth * hogResult.depth +
                       hogX_internal * hogResult.depth;
    
    //for(int pixelY = pixelY_start; pixelY < pixelY_end; pixelY++){
    //    for(int pixelX = pixelX_start; pixelX < pixelX_end; pixelX++){ 
    for(int offsetY = 1; offsetY < 2*sbin-1; offsetY++){ //as in voc-release5, only go from 1 to 2*(sbin-1), and skip the farther out pixels
        for(int offsetX = 1; offsetX < 2*sbin-1; offsetX++){

            //location in magImg and gradImg
            int pixelX = pixelX_start + offsetX;
            int pixelY = pixelY_start + offsetY; 
 
            //this pixel's contribution (weight) to our hog cell 
            float weightX = 1.0f - ((float)abs(sbin - offsetX + 0.5f) / sbin); //when offset=0, we're at -sbin from hog cell's center. when offset=2*sbin-1, we're +sbin from the center. the +0.5 is because we're indexing from top-left of cell, not from center of cell.
            float weightY = 1.0f - ((float)abs(sbin - offsetY + 0.5f) / sbin); // TODO: remove division by sbin 
            //printf("weightX = %f, weightY = %f \n", weightX, weightY);

            int oriBin_signed = (int)oriImg.at<float>(pixelY, pixelX); //TODO: just make oriImg a uchar img
            float mag = magImg.at<float>(pixelY, pixelX);

            //hogResult.hog[hogOutputIdx + oriBin_signed] += 1; //test
            //hogResult.hog[hogOutputIdx + oriBin_signed] += mag * 2.0f; //test
            hogResult.hog[hogOutputIdx + oriBin_signed] += mag * weightX * weightY;
        }
    }

    //TODO: calculate the sum of this bin and store it (for normalization)
}

//compute contrast-insensitive features (0 to 180 degrees)
// assume that hogCell() has already been performed on hogResult.
inline void PgHog::hogCell_unsigned(int hogX, int hogY, PgHogContainer hogResult){
    int hogX_internal = hogX + hogResult.padx; //skip over padding on left side of hogResult.hog
    int hogY_internal = hogY + hogResult.pady; //skip over padding at the top of hogResult.hog
    int hogOutputIdx = hogY_internal * hogResult.paddedWidth * hogResult.depth +
                       hogX_internal * hogResult.depth;

    //pool contrast-sensitive features to contrast-insensitive
    for(int i=0; i<9; i++){
        hogResult.hog[hogOutputIdx + 18 + i] = hogResult.hog[hogOutputIdx + 0 + i] + //0 to 180
                                               hogResult.hog[hogOutputIdx + 9 + i];  //180 to 360
    }
}

//compute the norm of one hog cell (sum up its 'energy'), store it in normImg
// assume that hogCell() has already been performed on hogResult.
inline void PgHog::hogCell_gradientEnergy(int hogX, int hogY, PgHogContainer hogResult, Mat &normImg){
    int hogX_internal = hogX + hogResult.padx; //skip over padding on left side of hogResult.hog
    int hogY_internal = hogY + hogResult.pady; //skip over padding at the top of hogResult.hog

    int hogIdx = hogY_internal * hogResult.paddedWidth * hogResult.depth +
                 hogX_internal * hogResult.depth;
 
    //sum up the (0 to 360 degree) hog cells
    float norm = 0.0f;
    for(int i=0; i<18; i++){
        norm += hogResult.hog[hogIdx];
    }
    normImg.at<float>(hogY_internal, hogX_internal) = norm;
}

//normalize HOG Cells (contrast-sensitive and contrast-insensitive features) into HOG Blocks
//produces *one* HOG Block based on (hogY, hogX) and its neighbors
inline void PgHog::hogBlock_normalize(int hogX, int hogY, PgHogContainer hogResult, Mat normImg){

    //TODO: calculate the normalization factors with convolution implementation? (sacrifice locality for reuse?)

    int hogX_internal = hogX + hogResult.padx; //skip over padding on left side of hogResult.hog
    int hogY_internal = hogY + hogResult.pady; //skip over padding at the top of hogResult.hog

    //TODO: clamp hogX_internal and hogY_internal to a minimum of 1, to avoid falling off the edge?

    //TODO: use const here (like ffld)? -- does const matter here?
    float n0 = 1 / sqrt(normImg.at<float>(hogY_internal-1, hogX_internal-1) + //top-left 
                        normImg.at<float>(hogY_internal-1, hogX_internal  ) + 
                        normImg.at<float>(hogY_internal  , hogX_internal-1) + 
                        normImg.at<float>(hogY_internal  , hogX_internal  ) + eps);

    float n1 = 1 / sqrt(normImg.at<float>(hogY_internal-1, hogX_internal  ) + //top-right
                        normImg.at<float>(hogY_internal-1, hogX_internal+1) + 
                        normImg.at<float>(hogY_internal  , hogX_internal  ) + 
                        normImg.at<float>(hogY_internal  , hogX_internal+1) + eps);

    float n2 = 1 / sqrt(normImg.at<float>(hogY_internal  , hogX_internal-1) + //bottom-left
                        normImg.at<float>(hogY_internal  , hogX_internal  ) + 
                        normImg.at<float>(hogY_internal+1, hogX_internal-1) + 
                        normImg.at<float>(hogY_internal+1, hogX_internal  ) + eps);

    float n3 = 1 / sqrt(normImg.at<float>(hogY_internal  , hogX_internal  ) + //bottom-right
                        normImg.at<float>(hogY_internal  , hogX_internal+1) + 
                        normImg.at<float>(hogY_internal+1, hogX_internal  ) + 
                        normImg.at<float>(hogY_internal+1, hogX_internal+1) + eps);


    int hogIdx = hogY_internal * hogResult.paddedWidth * hogResult.depth + 
                 hogX_internal * hogResult.depth;  //the location in hogResult.hog to update
    //contrast sensitive features (0 to 360 degrees)
    //  TODO

    //contrast-insensitive features (0 to 180 degrees)
    for(int i=0; i<9; i++){    
        //TODO: good candidate for vectorization, if compute bound?

        float currFeature = hogResult.hog[hogIdx + i + 18]; //contrast-insensitive feature in bin range 18 to 26
        float h0 = min(currFeature * n0, 0.2f); 
        float h1 = min(currFeature * n1, 0.2f);
        float h2 = min(currFeature * n2, 0.2f); 
        float h3 = min(currFeature * n3, 0.2f);

        hogResult.hog[hogIdx + i + 18] = (h0 + h1 + h2 + h3) * 0.5f; //TODO: check on numerical results 
    }
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
    //TODO: require padx>=1 and pady>=1. (is this enough to avoid the need for guards on block normalization?) 
    hogResult.width = round((float)img.cols / (float)spatialBinSize);
    hogResult.height = round((float)img.rows / (float)spatialBinSize);
    hogResult.paddedWidth = hogResult.width + 2*hogResult.padx;
    hogResult.paddedHeight = hogResult.height + 2*hogResult.pady;
    hogResult.depth = 32;
    hogResult.spatialBinSize = spatialBinSize;
    hogResult.hog = (float*)malloc(hogResult.paddedWidth * hogResult.paddedHeight * hogResult.depth * sizeof(float));
    
    //TODO: store normalization results
    //float* norm = malloc(hogResult.paddedWidth * hogResult.paddedHeight * sizeof(float)); 
    Mat normImg(hogResult.paddedHeight, hogResult.paddedWidth, CV_32FC1);

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
                hogCell(hogX-1, hogY-1, oriImg, magImg, hogResult); //constrast sensitive features
                hogCell_gradientEnergy(hogX-1, hogY-1, hogResult, normImg); //sum of each hog cell's contrast sensitive (0-360) bins
                hogCell_unsigned(hogX-1, hogY-1, hogResult); //contrast-insensitive features
            }

            //HOG block normalization
            //note: there are no 'if hogX>0' guards, because the hogResult.hog and normImg are padded.
            //      also, hogBlock_normalize() has a forward dependency to its right and bottom neighbors, so we do hogX-2, hogY-2
            hogBlock_normalize(hogX-2, hogY-2, hogResult, normImg); //TODO: think about edge cases
        }
    }

    //writeGradToFile(oriImg, magImg);
    writeHogCellsToFile(hogResult);
}

//----------------- TEMP DEBUG functions below this line ------------------

void writeGradToFile(Mat oriImg, Mat magImg){
    oriImg.convertTo(oriImg, CV_8UC1, 255.);
    imwrite("PgHog_orientations.jpg", oriImg);
    
    //magImg.convertTo(magImg, CV_8UC1, 255.);
    imwrite("PgHog_magnitudes.jpg", magImg); 
}

void writeHogCellsToFile(PgHogContainer hogResult){
    ostringstream fname;
    fname << "piggyHOG_results/level" << 0 << ".csv";
    writeCsv_3d_Hog_Float(hogResult.hog, hogResult.paddedWidth, hogResult.paddedHeight, hogResult.depth, fname.str());
}

