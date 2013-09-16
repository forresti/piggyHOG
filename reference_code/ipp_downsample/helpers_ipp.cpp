
#include "helpers.h"
using namespace std;

void forrestWritePgm(cv::Mat img, std::string out_filename)
{
    //require a 32-bit float or 8-bit uchar with 1 or 3 channels.
    assert( (img.type() == CV_8UC3) || (img.type() == CV_32FC3) || (img.type() == CV_8UC1) || (img.type() == CV_32FC1));

    if((img.type() == CV_32FC3))
    {
        //PNG likes unsigned chars, 0-255.
        img.convertTo(img, CV_8UC3, 255.);
    }

    if((img.type() == CV_32FC1))
    {
        //PNG likes unsigned chars, 0-255.
        img.convertTo(img, CV_8UC1, 255.);
    }

    vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_PXM_BINARY); //PXM_BINARY is for PGM and PPM images
    compression_params.push_back(9);
    cv::imwrite(out_filename, img, compression_params);
}

//from parlab bootcamp 2012 code
double read_timer()
{
    static bool initialized = false;
    static struct timeval start;
    struct timeval end;
    if( !initialized )
    {
        gettimeofday( &start, NULL );
        initialized = true;
    }
    gettimeofday( &end, NULL );
    return (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);
}

