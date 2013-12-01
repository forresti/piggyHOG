#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cassert>
#include <xmmintrin.h>
#include <pmmintrin.h> //for _mm_hadd_pd()

#include "SimpleImg.h"
#include "streamHog.h"
#include "helpers.h"
using namespace std;


int main (int argc, char **argv)
{
    //init_lookup_table(); //similar to FFLD hog
    //init_atan2_constants(); //easier to vectorize alternative to lookup table. (similar to VOC5 hog)

    streamHog sHog;

    int ALIGN_IN_BYTES = 256;
    int n_iter = 1000; //not really "iterating" -- just number of times to run the experiment
    //int stride = width + (ALIGN_IN_BYTES - width%ALIGN_IN_BYTES); //thanks: http://stackoverflow.com/questions/2403631
    //SimpleImg img(height, width, stride, n_channels);

    SimpleImg img("../../images_640x480/carsgraz_001.image.jpg");

    //[mag, ori] = gradient_wideload_unvectorized(img)
    SimpleImg mag(img.height, img.width, img.stride, 1); //out img has just 1 channel
    SimpleImg ori(img.height, img.width, img.stride, 1); //out img has just 1 channel
    double start_timer = read_timer();
    for(int i=0; i<n_iter; i++){
        //gradient_wideload_unvectorized(img.height, img.width, img.stride, img.n_channels, ori.n_channels, img.data, ori.data, mag.data); 
        sHog.gradient_sse(img.height, img.width, img.stride, img.n_channels, ori.n_channels, img.data, ori.data, mag.data); 
    }
    mag.simple_csvwrite("mag.csv");
    mag.simple_imwrite("mag.jpg");
    ori.simple_imwrite("ori.jpg");

    double stream_time = (read_timer() - start_timer) / n_iter;
    double gb_to_copy = img.width * img.height * img.n_channels * sizeof(pixel_t) / 1e9;
    double gb_per_sec = gb_to_copy / (stream_time/1000); //convert stream_time from ms to sec
    printf("avg stream time = %f ms, %f GB/s \n", stream_time, gb_per_sec);

    if(n_iter < 100){
        printf("WARNING: n_iter = %d. For statistical significance, we recommend n_iter=100 or greater. \n", n_iter);
    }

    return 0;
}


