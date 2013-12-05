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


//test template for: 
//  bestChannel = argmax(mag[0,1,2]); 
//  [gradX, gradY] = gradX[bestChannel], gradY[bestChannel]
//@in-out gradX_max[8], gradY_max[8] = result from this channel's iteration
//@return bool = "pass/fail"
bool test_ori_argmax(int16_t magChannel[8],    int16_t old_magMax[8],
                    int16_t gradX_channel[8], int16_t gradY_channel[8],
                    int16_t* gradX_max,     int16_t* gradY_max,
                    int16_t gold_gradX_max[8], int16_t gold_gradY_max[8]) //gold_* = expected output
{

    streamHog shog;

    //copy to SSE registers
    __m128i magChannel_sse = _mm_load_si128((__m128i*)magChannel); //current channel's mag
    __m128i old_magMax_sse = _mm_load_si128((__m128i*)old_magMax); //max mag found in previous channels
    __m128i gradX_channel_sse = _mm_load_si128((__m128i*)gradX_channel); //current channel's grad{X,Y}
    __m128i gradY_channel_sse = _mm_load_si128((__m128i*)gradY_channel);
    __m128i gradX_max_sse = _mm_load_si128((__m128i*)gradX_max);
    __m128i gradY_max_sse = _mm_load_si128((__m128i*)gradY_max);

    shog.select_epi16(magChannel_sse, old_magMax_sse, 
                      gradX_channel_sse, gradY_channel_sse,
                      gradX_max_sse, gradY_max_sse); //grad{X,Y}_max_sse are passed by ref, so they get updated.
   
    //print_epi16(gradY_max_sse, "gradY_max"); //TODO: get this to work w/o segfault
    //print_epi16(gradY_max_sse, "gradY_max");
 
    //copy back passed-by-ref grad{X,Y}_max_sse
    _mm_store_si128((__m128i*)gradX_max, gradX_max_sse); //write SSE result back to scalar grad{X,Y}_max
    _mm_store_si128((__m128i*)gradY_max, gradY_max_sse);

    //check correctness
    bool isGood = true;
    for(int i=0; i<8; i++){
        if(gradY_max[i] != gold_gradX_max[i]){
            isGood = false;
            printf("    gradX_max[%d]. expected:%d, got:%d \n", i, gold_gradX_max[i], gradX_max[i]);
        }
        if(gradY_max[i] != gold_gradY_max[i]){
            isGood = false;
            printf("    gradY_max[%d]. expected:%d, got:%d \n\n", i, gold_gradY_max[i], gradY_max[i]);
        }
    }
    return isGood;
}

//scalar reference impl.
//you call this ONCE per channel. you externally update old_magMax.
//  TODO: move reference_ori_argmax() to streamHog.cpp
//@in-out gradX_max[8], gradY_max[8]
void reference_ori_argmax(int16_t magChannel[8],    int16_t old_magMax[8],
                          int16_t gradX_channel[8], int16_t gradY_channel[8],
                          int16_t* gradX_max,     int16_t* gradY_max)
{

    for(int i=0; i<8; i++){ //iterate over sse-style vector
        if(magChannel[i] > old_magMax[i]){
            gradX_max[i] = gradX_channel[i];
            gradY_max[i] = gradY_channel[i];    
        }
    }
    //now, YOU update old_magMax and call again on next channel.
}

//test streamHog::select_epi16(), which gets the argmax mag channel, 
//     and stores the gradX,gradY of that argmax channel.
bool run_tests_ori_argmax(){
    int numFailed=0;

  //test suite with 3 subtests; 1 per channel
    int16_t  magChannel0[8] = {1,2,3,4,5,6,7,8};
    int16_t  magChannel1[8] = {110,120,130,140,150,160,1,180};
    int16_t  magChannel2[8] = {0,200,50,0,0,0,0,0};
    int16_t* magChannel[3] = {magChannel0, magChannel1, magChannel2}; magChannel[3][8];
    int16_t  old_magMax[8] = {0,0,0,0,0,0,0,0};   

    int16_t gradX_channel[3][8]; 
    int16_t gradY_channel[3][8];
    int16_t gradX_max[8] = {0,0,0,0,0,0,0,0};
    int16_t gradY_max[8] = {0,0,0,0,0,0,0,0};

    //initialize gradX, gradY with some arbitrary values...
    // gradX[ch=0][:] = 1,...,8. gradX[ch=1][:] = 11...18, gradX[ch=2][:] = 21...28 
    for(int ch=0; ch<3; ch++){
        for(int i=0; i<8; i++){
            gradX_channel[ch][i] = ch*10 + i + 1;
            gradY_channel[ch][i] = ch*10 + i + 1;
        }
    }
    
    int16_t gold_gradX_max[8];
    int16_t gold_gradY_max[8];

    //test grad{X,Y}_max calculation. (this is the gradX, gradY of the argmax magnitude channel)
    for(int ch=0; ch<3; ch++){
        printf("channel %d \n", ch);

        //calculate 'gold' for this channel's iteration
        reference_ori_argmax(magChannel[ch], old_magMax, 
                             gradX_channel[ch], gradY_channel[ch],
                             gold_gradX_max, gold_gradY_max); //updates gold_grad{X,Y}_max  

        bool isGood = test_ori_argmax(magChannel[ch], old_magMax, 
                                      gradX_channel[ch], gradY_channel[ch], //TODO: &gradX_channel[ch][0], if needed
                                      gradX_max, gradY_max,
                                      gold_gradX_max, gold_gradY_max);

        for(int i=0; i<8; i++){ //emulate _mm_max_epi16(magChannel[ch], old_magMax)
            old_magMax[i] = max(magChannel[ch][i], old_magMax[i]); 
        }
        if(!isGood){ numFailed++; }
   }

    printf("number of select_epi16 tests failed: %d \n", numFailed); 
}

//TODO: put this into a class (like PgHogContainer or streamHog), once I decide what data types to use
//@output-by-ref out_hogWidth out_hogHeight
//@return hogWidth = memory aligned vector for storing HOG histogram 
float* allocate_hist(int in_imgHeight, int in_imgWidth, int sbin,
                   int &out_hogHeight, int &out_hogWidth){

    out_hogHeight = round(in_imgHeight/sbin);
    out_hogWidth = round(in_imgWidth/sbin);
    const int hogDepth = 32;

    float* hogBuffer = (float*)malloc_aligned(32, out_hogWidth * out_hogHeight * hogDepth * sizeof(float));
    memset(hogBuffer, 0, out_hogWidth * out_hogHeight * hogDepth * sizeof(float));
    return hogBuffer;
} 

// MAIN TEST OF FUNCTIONALITY
void test_computeCells_voc5_vs_streamHOG(){
    streamHog sHog; //streamHog constructor initializes lookup tables & constants (mostly for orientation bins)
    int sbin = 4;

    SimpleImg img("../../images_640x480/carsgraz_001.image.jpg");
    SimpleImg ori(img.height, img.width, img.stride, 1); //out img has just 1 channel
    SimpleImg mag(img.height, img.width, img.stride, 1); //out img has just 1 channel
    int hogWidth, hogHeight;
    float* hogBuffer_voc5 = allocate_hist(img.height, img.width, sbin,
                                          hogHeight, hogWidth); //hog{Height,Width} are passed by ref.
    float* hogBuffer_streamHog = allocate_hist(img.height, img.width, sbin,
                                               hogHeight, hogWidth); //hog{Height,Width} are passed by ref.

  //[mag, ori] = gradient_sse(img)
    sHog.gradient_sse(img.height, img.width, img.stride, img.n_channels, ori.n_channels, img.data, ori.data, mag.data); 
    //sHog.gradient_voc5_reference(img.height, img.width, img.stride, img.n_channels, ori.n_channels, img.data, ori.data, mag.data);

  //hist = computeCells(mag, ori, sbin)
    sHog.computeCells_voc5_reference(img.height, img.width, img.stride, sbin,
                                     ori.data, mag.data, 
                                     hogHeight, hogWidth, hogBuffer_voc5); 
    sHog.computeCells_stream(img.height, img.width, img.stride, sbin,
                             ori.data, mag.data,
                             hogHeight, hogWidth, hogBuffer_streamHog);

    int hogDepth = 32;
    float eps_diff = 0.01;

    //check if it matches...
    for(int y=0; y<hogHeight; y++){
        for(int x=0; x<hogWidth; x++){
            for(int d=0; d<hogDepth; d++){
                float voc5_element = hogBuffer_voc5[x*hogDepth + y*hogWidth*hogDepth + d];
                float streamHog_element = hogBuffer_streamHog[x*hogDepth + y*hogWidth*hogDepth + d];
                if( fabs(streamHog_element - voc5_element) > eps_diff){
                    printf("x=%d, y=%d, d=%d. voc5=%f, streamHog=%f \n", x, y, d, voc5_element, streamHog_element);
                }
            }
        }
    }
}

// MAIN TEST OF FUNCTIONALITY
void test_streamHog_oneScale(){
    streamHog sHog; //streamHog constructor initializes lookup tables & constants (mostly for orientation bins)

    int sbin = 4;

    int ALIGN_IN_BYTES = 256;
    int n_iter = 200; //not really "iterating" -- just number of times to run the experiment
    if(n_iter < 100){
        printf("WARNING: n_iter = %d. For statistical significance, we recommend n_iter=100 or greater. \n", n_iter);
    }
    //int stride = width + (ALIGN_IN_BYTES - width%ALIGN_IN_BYTES); //thanks: http://stackoverflow.com/questions/2403631
    //SimpleImg img(height, width, stride, n_channels);

    SimpleImg img("../../images_640x480/carsgraz_001.image.jpg");
    SimpleImg ori(img.height, img.width, img.stride, 1); //out img has just 1 channel
    SimpleImg mag(img.height, img.width, img.stride, 1); //out img has just 1 channel
    int hogWidth, hogHeight;
    float* hogBuffer = allocate_hist(img.height, img.width, sbin,
                                     hogHeight, hogWidth); //hog{Height,Width} are passed by ref.

  //[mag, ori] = gradient_sse(img)
    double start_timer = read_timer();
    for(int i=0; i<n_iter; i++){
        sHog.gradient_sse(img.height, img.width, img.stride, img.n_channels, ori.n_channels, img.data, ori.data, mag.data); 
        //sHog.gradient_voc5_reference(img.height, img.width, img.stride, img.n_channels, ori.n_channels, img.data, ori.data, mag.data);
    }

    double stream_time = (read_timer() - start_timer) / n_iter;
    double gb_to_copy = img.width * img.height * img.n_channels * sizeof(pixel_t) / 1e9;
    double gb_per_sec = gb_to_copy / (stream_time/1000); //convert stream_time from ms to sec
    printf("avg (mag, ori) stream time = %f ms, %f GB/s \n", stream_time, gb_per_sec);

    //mag.simple_csvwrite("mag.csv");
    mag.simple_imwrite("mag.jpg");
    ori.simple_imwrite("ori.jpg");

  //hist = computeCells(mag, ori, sbin)
    start_timer = read_timer();

    //TODO: figure out how to test correctness: voc5 vs streamHog computeCells().
    for(int i=0; i<n_iter; i++){
        //ori and mag both have size {img.height, img.width}
 
//        sHog.computeCells_voc5_reference(img.height, img.width, img.stride, sbin,
//                                         ori.data, mag.data, 
//                                         hogHeight, hogWidth, hogBuffer); 

        sHog.computeCells_stream(img.height, img.width, img.stride, sbin,
                                 ori.data, mag.data,
                                 hogHeight, hogWidth, hogBuffer);
    }

    stream_time = (read_timer() - start_timer) / n_iter;
    int hogDepth = 32;
    gb_to_copy = hogWidth * hogHeight * hogDepth * sizeof(float) / 1e9; //TODO: change 'float' to new data type if necessary
    gb_per_sec = gb_to_copy / (stream_time/1000); //convert stream_time from ms to sec
    printf("avg hogCell stream time = %f ms, %f GB/s \n", stream_time, gb_per_sec);

}

int main (int argc, char **argv)
{
    //run_tests_ori_argmax(); //unit test
    //test_computeCells_voc5_vs_streamHOG(); //unit test
    test_streamHog_oneScale(); //timing experiment

    return 0;
}


