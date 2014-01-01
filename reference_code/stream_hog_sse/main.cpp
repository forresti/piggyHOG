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
using namespace std;

int main (int argc, char **argv)
{
    //run_tests_ori_argmax(); //unit test
    test_computeCells_voc5_vs_streamHOG(); //unit test
    test_streamHog_oneScale(); //timing experiment

    return 0;
}

