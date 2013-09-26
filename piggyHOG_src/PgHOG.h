#ifndef __PGHOG_H__
#define __PGHOG_H__
#include <opencv2/opencv.hpp>
#include "PgHogContainer.h"

using namespace std;
using namespace cv;

//fast HOG feature extraction
// complies with the UOC-TTI Deformable Parts Model HOG extractor

class PgHOG{

    public:

    //precompute tables, stuff like that
    PgHOG(); 
    ~PgHOG();

    //compute HOGs at many scales of downsampling
    vector<PgHogContainer> extract_HOG_pyramid(Mat img, int spatialBinSize); //in voc-release5, spatialBinSize is typically 4 or 8 pixels

    PgHogContainer extract_HOG_oneScale(Mat img, int spatialBinSize);

    //TODO: PgHOGContainer struct {float* hog; int width; int height; int depth}
    //      perhaps put this in PgHOGContainer.h

};

#endif

