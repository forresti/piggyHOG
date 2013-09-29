#include "helpers.h"

using namespace std;

void writeCsv_2dFloat(const float* vec, int nRows, int nCols, string fname)
{
    ofstream myfile;
    myfile.open(fname.c_str());
    for(int row=0; row<nRows; row++){
        for(int col=0; col<(nCols-1); col++){
            myfile << vec[row*nCols + col] << ",";
        }
        myfile << vec[row*nCols + (nCols-1)] << "\n"; //don't put a ',' after final element on a CSV line
    }
    myfile.close();
}

void writeCsv_3d_Hog_Float(const float* vec, int width, int height, int depth, string fname)
{
    ofstream myfile;
    myfile.open(fname.c_str());
    myfile << depth << "," << width << "," << height << "\n"; //dimensions

    for(int y=0; y<height; y++){
        for(int x=0; x<width; x++){
            for(int d=0; d<depth; d++){
                myfile << vec[y*width*depth + x*depth + d] << ",";
            }
            myfile << vec[y*width*depth + x*depth + (depth-1)] << "\n";
        }
    }
    myfile.close();
}

double read_timer(){
    struct timeval start;
    gettimeofday( &start, NULL );
    return (double)((start.tv_sec) + 1.0e-6 * (start.tv_usec)) * 1000; //in milliseconds
}

