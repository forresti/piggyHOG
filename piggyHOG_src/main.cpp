#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "common/helpers.h"
#include "PgHog.h"
using namespace std;

void writeHogCellsToFile(vector<PgHogContainer*> hogPyramid){
    for(int level = 0; level < hogPyramid.size(); level++){
        ostringstream fname;
        fname << "piggyHOG_results/level" << level << ".csv";
        writeCsv_3d_Hog_Float(hogPyramid[level]->hog, hogPyramid[level]->paddedWidth, hogPyramid[level]->paddedHeight, hogPyramid[level]->depth, fname.str());
    }
}

void test_oneLevel(){
    Mat img = imread("../images_640x480/carsgraz_001.image.jpg"); //OpenCV 8U_C3 image
    PgHog pghog;

    int n_iter = 10;
    int spatialBinSize = 4;

    double start_timer = read_timer();
    for(int i=0; i<n_iter; i++){
        PgHogContainer* hogResult = pghog.extract_HOG_oneScale(img, spatialBinSize);
    }
    double time_one_scale = (read_timer() - start_timer)/n_iter;

    printf("time for 1 scale: %f ms \n", time_one_scale);
}

void test_pyramid(){
    Mat img = imread("../images_640x480/carsgraz_001.image.jpg"); //OpenCV 8U_C3 image
    PgHog pghog;
    int spatialBinSize = 4;

    double start_timer = read_timer();
    vector<PgHogContainer*> hogPyramid = pghog.extract_HOG_pyramid(img, 11, 6);
    double time_one_scale = read_timer() - start_timer;

    printf("time for hog pyramid: %f ms \n", time_one_scale);
    //writeHogCellsToFile(hogPyramid);
}

int main (int argc, char **argv)
{
    test_oneLevel();
//    test_pyramid();

    return 0;
}

