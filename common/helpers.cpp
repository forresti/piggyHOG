#include "helpers.h"

using namespace std;

void writeCsv_2dFloat(vector<float> vec, int nRows, int nCols, string fname)
{
    ofstream myfile;
    myfile.open(fname.c_str());
    for(int row=0; row<nRows; row++)
    {
        for(int col=0; col<(nCols-1); col++)
        {
            myfile << vec[row*nCols + col] << ",";
        }
        myfile << vec[row*nCols + (nCols-1)] << endl; //don't put a ',' after final element on a CSV line
    }
    myfile.close();
}



