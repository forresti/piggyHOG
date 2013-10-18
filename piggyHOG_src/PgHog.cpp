#include "PgHog.h"
#include <iomanip>
#include <iostream>
#include <fstream>
#include "omp.h"
using namespace std;
using namespace cv;

#define eps 0.0001

static inline int clamp(int idx, int min_idx, int max_idx){
    return max(min_idx, min(idx, max_idx));
}

//use OpenCV's bilinear filter downsampling
Mat downsampleWithOpenCV(Mat img, double scale){ //TODO: move this to 'helpers' file
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

PgHog::PgHog(){

    // Fill the atan2 table (from FFLD) 
    //if (ATAN2_TABLE[0][0] == 0) {
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
    //}

}

PgHog::~PgHog(){

}

//temp debug function declarations
void writeGradToFile(Mat oriImg, Mat gradImg);
//void writeHogCellsToFile(PgHogContainer hogResult);

//compute the gradient and magnitude at one image location, store the results in oriImg and magImg
inline void PgHog::gradient(int x, int y, Mat img, Mat &oriImg, Mat &magImg){
    x = clamp(x, 1, img.cols-1);
    y = clamp(y, 1, img.rows-1);

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
    //float ori = cv::fastAtan2((double)gradY, (double)gradX);
    float ori = ATAN2_TABLE[(int)gradY + 255][(int)gradX + 255]; //these are already scaled to range of 0-18
    max_mag = sqrt(max_mag); //we've been using magnitude-squared so far

    oriImg.at<float>(y, x) = ori;
    magImg.at<float>(y, x) = max_mag;
}

//compute one HOG cell, storing the results in hogResult
// only compute the contrast-sensitive features (0 to 360 degrees)
inline void PgHog::hogCell(int hogX, int hogY, Mat &oriImg, Mat &magImg, PgHogContainer* hogResult){
    //populate this HOG cell by linearly interpolating the oriented gradients 
    //the 'center' of the hog cell is: (hogX+sbin/2, hogY+sbin/2).
    //we do (x,y)=(+/-sbin, +/-sbin) pixels from the center of the hog cell.

    //const int sbin=4;
    int sbin = hogResult->spatialBinSize;
    int hogX_internal = hogX + hogResult->padx; //skip over padding on left side of hogResult->hog
    int hogY_internal = hogY + hogResult->pady; //skip over padding at the top of hogResult->hog

    int pixelX_start = hogX*sbin - sbin*0.5f;
    pixelX_start = clamp(pixelX_start, 0, magImg.cols-1); //not exactly the right logic ... should actually skip the indices that fall off the edge, instead of clamping
    //int pixelX_end = pixelX_start + 2*sbin;    
    
    int pixelY_start = hogY*sbin - sbin*0.5f;
    pixelY_start = clamp(pixelY_start, 0, magImg.rows-1);
    //int pixelY_end = pixelY_start + 2*sbin;

    int hogOutputIdx = hogY_internal * hogResult->paddedWidth * hogResult->depth +
                       hogX_internal * hogResult->depth;
    
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

            int oriBin_signed = (int)oriImg.at<float>(pixelY, pixelX); //TODO: just make oriImg a uchar img
            float mag = magImg.at<float>(pixelY, pixelX);

            hogResult->hog[hogOutputIdx + oriBin_signed] += mag * weightX * weightY;
        }
    }
}

//compute contrast-insensitive features (0 to 180 degrees)
// assume that hogCell() has already been performed on hogResult->
inline void PgHog::hogCell_unsigned(int hogX, int hogY, PgHogContainer* hogResult){
    int hogX_internal = hogX + hogResult->padx; //skip over padding on left side of hogResult->hog
    int hogY_internal = hogY + hogResult->pady; //skip over padding at the top of hogResult->hog
    int hogOutputIdx = hogY_internal * hogResult->paddedWidth * hogResult->depth +
                       hogX_internal * hogResult->depth;

    //pool contrast-sensitive features to contrast-insensitive
    for(int i=0; i<9; i++){
        hogResult->hog[hogOutputIdx + 18 + i] = hogResult->hog[hogOutputIdx + 0 + i] + //0 to 180
                                               hogResult->hog[hogOutputIdx + 9 + i];  //180 to 360
    }
}

//compute the norm of one hog cell (sum up its 'energy'), store it in normImg
// assume that hogCell() has already been performed on hogResult->
inline void PgHog::hogCell_gradientEnergy(int hogX, int hogY, PgHogContainer* hogResult, Mat &normImg){
    int hogX_internal = hogX + hogResult->padx; //skip over padding on left side of hogResult->hog
    int hogY_internal = hogY + hogResult->pady; //skip over padding at the top of hogResult->hog

    int hogIdx = hogY_internal * hogResult->paddedWidth * hogResult->depth +
                 hogX_internal * hogResult->depth;
 
    //sum up the (0 to 360 degree) hog cells
    float norm = 0.0f;
    for(int i=0; i<18; i++){
        norm += hogResult->hog[hogIdx+i] * hogResult->hog[hogIdx+i]; //squared -- will do sqrt in hogBlock_normalize()
    }
    normImg.at<float>(hogY_internal, hogX_internal) = norm;
}

//normalize HOG Cells (contrast-sensitive and contrast-insensitive features) into HOG Blocks
//produces *one* HOG Block based on (hogY, hogX) and its neighbors
inline void PgHog::hogBlock_normalize(int hogX, int hogY, PgHogContainer* hogResult, Mat normImg){

    //TODO: calculate the normalization factors with convolution implementation? (sacrifice locality for reuse?)

    int hogX_internal = hogX + hogResult->padx; //skip over padding on left side of hogResult->hog
    int hogY_internal = hogY + hogResult->pady; //skip over padding at the top of hogResult->hog

    //TODO: clamp hogX_internal and hogY_internal to a minimum of 1, to avoid falling off the edge?

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


    int hogIdx = hogY_internal * hogResult->paddedWidth * hogResult->depth + 
                 hogX_internal * hogResult->depth;  //the location in hogResult->hog to update
    
    float t0 = 0.0f; //for texture features
    float t1 = 0.0f;
    float t2 = 0.0f; 
    float t3 = 0.0f; 

    //contrast sensitive features (0 to 360 degrees)
    for(int i=0; i<18; i++){
        float currFeature = hogResult->hog[hogIdx + i]; //contrast-sensitive feature in bin range 0 to 17
        float h0 = min(currFeature * n0, 0.2f);
        float h1 = min(currFeature * n1, 0.2f);
        float h2 = min(currFeature * n2, 0.2f);
        float h3 = min(currFeature * n3, 0.2f);

        hogResult->hog[hogIdx + i] = (h0 + h1 + h2 + h3) * 0.5f; //TODO: check on numerical results 

        t0 += h0; //precompute texture features
        t1 += h1;
        t2 += h2;
        t3 += h3;
    }

    //contrast-insensitive features (0 to 180 degrees)
    for(int i=0; i<9; i++){    
        //TODO: good candidate for vectorization, if compute bound?

        float currFeature = hogResult->hog[hogIdx + i + 18]; //contrast-insensitive feature in bin range 18 to 26
        float h0 = min(currFeature * n0, 0.2f); 
        float h1 = min(currFeature * n1, 0.2f);
        float h2 = min(currFeature * n2, 0.2f); 
        float h3 = min(currFeature * n3, 0.2f);

        hogResult->hog[hogIdx + i + 18] = (h0 + h1 + h2 + h3) * 0.5f; //TODO: check on numerical results 
    }

    //texture features
    hogResult->hog[hogIdx + 27] = t0 * 0.2357f; //FIXME: some of these are greater than 0.4, seems wrong.
    hogResult->hog[hogIdx + 28] = t1 * 0.2357f;
    hogResult->hog[hogIdx + 29] = t2 * 0.2357f;
    hogResult->hog[hogIdx + 30] = t3 * 0.2357f;
}

//add binary place features around padded border of a HOG image
inline void PgHog::hogPlaceFeatures_border(PgHogContainer* hogResult){

    for(int hogY=0; hogY < hogResult->pady; hogY++){
        for(int hogX=0; hogX < hogResult->padx; hogX++){
            int topLeftIdx =      hogX                     + (hogY * hogResult->paddedWidth);
            int topRightIdx =    (hogX + hogResult->width) + (hogY * hogResult->paddedWidth);
            int bottomLeftIdx =   hogX                     + ((hogY + hogResult->height) * hogResult->paddedWidth);
            int bottomRightIdx = (hogX + hogResult->width) + ((hogY + hogResult->height) * hogResult->paddedWidth);
            
            hogResult->hog[topLeftIdx]     = 1.0f;
            hogResult->hog[topRightIdx]    = 1.0f;
            hogResult->hog[bottomLeftIdx]  = 1.0f;
            hogResult->hog[bottomRightIdx] = 1.0f;
        }
    }
}

PgHogContainer* PgHog::extract_HOG_oneScale(Mat img, int spatialBinSize){
  //setup
    assert(img.type() == CV_8UC3);
    int sbin = spatialBinSize; //shorthand

    //TODO: possibly go down to 8-bit char
    Mat oriImg(img.rows, img.cols, CV_32FC1); //TODO: replace with aligned mem?
    Mat magImg(img.rows, img.cols, CV_32FC1);

    //hogResult first holds HOG Cells, then is normalized into HOG Blocks.
    PgHogContainer *hogResult = (PgHogContainer*)malloc(sizeof(PgHogContainer)); //TODO: make PgHogContainer a class, and use new/delete.
    hogResult->padx = 11; //temporary 
    hogResult->pady = 6;
    //TODO: require padx>=1 and pady>=1. (is this enough to avoid the need for guards on block normalization?) 
    hogResult->width = round((float)img.cols / (float)spatialBinSize);
    hogResult->height = round((float)img.rows / (float)spatialBinSize);
    hogResult->paddedWidth = hogResult->width + 2*hogResult->padx;
    hogResult->paddedHeight = hogResult->height + 2*hogResult->pady;
    hogResult->depth = 32;
    hogResult->spatialBinSize = spatialBinSize;
    hogResult->hog = (float*)calloc(hogResult->paddedWidth * hogResult->paddedHeight * hogResult->depth, sizeof(float));    

    Mat normImg(hogResult->paddedHeight, hogResult->paddedWidth, CV_32FC1); //store normalization results

    const int unrollX = 8;
    const int unrollY = 4;
  //extract features
    //for(int hogY = 0; hogY < hogResult->height; hogY++){
    //    for(int hogX = 0; hogX < hogResult->width; hogX++){

    for(int hogY_tile = 0; hogY_tile < hogResult->height; hogY_tile += unrollY){
      for(int hogX_tile = 0; hogX_tile < hogResult->width; hogX_tile += unrollX){

        for(int hogY_inner = 0; hogY_inner < unrollY; hogY_inner++){
          for(int hogX_inner = 0; hogX_inner < unrollX; hogX_inner++){ 
            
            int hogX = hogX_tile + hogX_inner;
            int hogY = hogY_tile + hogY_inner;

            if(hogX >= hogResult->width || hogY >= hogResult->height)
                continue;

            //calculate gradients 
            for(int y=0; y<spatialBinSize; y++){ //TODO: move these loops into PgHog::gradient()?
                for(int x=0; x<spatialBinSize; x++){
                    //update oriImg and magImg at this x,y location
                    PgHog::gradient(hogX*sbin + x, hogY*sbin + y, img, oriImg, magImg);  
                }
            }

#if  1
            //HOG cell binning
            if(hogX>0 && hogY>0){
                hogCell(hogX-1, hogY-1, oriImg, magImg, hogResult); //hog[0:17] = constrast sensitive features
                hogCell_gradientEnergy(hogX-1, hogY-1, hogResult, normImg); //normImg = sum of each hog cell's contrast sensitive (0-360) bins
                hogCell_unsigned(hogX-1, hogY-1, hogResult); //hog[18:27] = contrast-insensitive features
            }

            //HOG block normalization
            //note: there are no 'if hogX>0' guards, because the hogResult->hog and normImg are padded.
            //      also, hogBlock_normalize() has a forward dependency to its right and bottom neighbors, so we do hogX-2, hogY-2
            hogBlock_normalize(hogX-2, hogY-2, hogResult, normImg); //TODO: think about edge cases
#endif
          }
        }
      }
    }

    //clean up rightmost columns
    for(int hogY = 0; hogY < hogResult->height+2; hogY++){ 
        for(int hogX = hogResult->width; hogX < hogResult->width+2; hogX++){
            hogBlock_normalize(hogX-2, hogY-2, hogResult, normImg);
        }
    }
    //cleanup bottom rows
    for(int hogY = hogResult->height; hogY < hogResult->height+2; hogY++){
        for(int hogX = 0; hogX < hogResult->width+2; hogX++){
            hogBlock_normalize(hogX-2, hogY-2, hogResult, normImg);
        }
    }

    hogPlaceFeatures_border(hogResult); //binary truncation features

    //writeGradToFile(oriImg, magImg);
    //writeHogCellsToFile(hogResult);
    return hogResult;
}

vector<PgHogContainer*> PgHog::extract_HOG_pyramid(Mat img, int padx, int pady){

    int interval = 10;
    float sc = pow(2, 1 / (float)interval);
    vector<Mat> imgPyramid(interval*2); //100% down to 25% of orig size (two octaves, 10 scales per octave)
    int nLevels = 30; //TODO: compute this based on img size
    vector<PgHogContainer*> hogPyramid(nLevels); //do I need a copy constructor for PgHogContainer?

//TODO: pass padx, pady into extract_HOG_oneScale()    

    //omp_set_num_threads(5); //hmm, default thread count seems best
    #pragma omp parallel for
    for(int i=0; i<interval; i++){
        float downsampleFactor = 1/pow(sc, i);
        //printf("downsampleFactor = %f \n", downsampleFactor);

        imgPyramid[i] = downsampleWithOpenCV(img, downsampleFactor);
        imgPyramid[i + interval] = downsampleWithOpenCV(img, downsampleFactor/2);

        hogPyramid[i] = extract_HOG_oneScale(imgPyramid[i], 4); //sbin=4
        hogPyramid[i + interval] = extract_HOG_oneScale(imgPyramid[i], 8); //sbin=8
        hogPyramid[i + 2*interval] = extract_HOG_oneScale(imgPyramid[i + interval], 8);

        //TODO: more small pyra levels?    

    }

    return hogPyramid;
}

//----------------- TEMP DEBUG functions below this line ------------------

void writeGradToFile(Mat oriImg, Mat magImg){
    oriImg.convertTo(oriImg, CV_8UC1, 255.);
    imwrite("PgHog_orientations.jpg", oriImg);
    
    //magImg.convertTo(magImg, CV_8UC1, 255.);
    imwrite("PgHog_magnitudes.jpg", magImg); 
}

#if 0
void writeHogCellsToFile(PgHogContainer hogResult){
    ostringstream fname;
    fname << "piggyHOG_results/level" << 0 << ".csv";
    writeCsv_3d_Hog_Float(hogResult->hog, hogResult->paddedWidth, hogResult->paddedHeight, hogResult->depth, fname.str());
}
#endif
