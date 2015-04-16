#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cassert>
#include <xmmintrin.h>
#include <pmmintrin.h> //for _mm_hadd_pd()

//#include "SimpleImg.h"
#include "SimpleImg.hpp"
#include "streamHog.h"
#include "helpers.h"
#include "test_streamHog.h"
#include "driver_streamHog.h"

using namespace std;

int main (int argc, char **argv)
{
    //run_tests_ori_argmax(); //unit test
//    test_computeCells_voc5_vs_streamHOG(); //unit test -- computeCells_stream matches voc5. computeCells_stream_noBorderCheck matches except for ~100 pixels.

//    test_streamHog_oneScale_default(); //timing experiment
//    test_streamHog_pyramid(); //timing experiment

//    streamHog_pyramid(); //run the pyramid for real
    outerLoopParallel_streamHog_pyramid();
 
    return 0;
}

