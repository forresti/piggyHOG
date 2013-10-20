#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "helpers.h"
using namespace std;
using namespace cv;


//template <class pixelType>
void stream_simple(ForrestImg img){


}


int main (int argc, char **argv)
{
    int n_iter = 10;

    double start_timer = read_timer();
    double stream_time = (read_timer() - start_timer) / n_iter;
    printf("avg stream time = %f ms \n", stream_time);


    return 0;
}


