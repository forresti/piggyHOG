#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cassert>
#include <xmmintrin.h>
#include <pmmintrin.h> //for _mm_hadd_pd()
#include <opencv2/opencv.hpp> //only used for file I/O

//#include "SimpleImg.h"
#include "SimpleImg.hpp"
#include "streamHog.h"
#include "helpers.h"
#include "test_streamHog.h"
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

template<class my_pixel_t>
void diff_imgs(my_pixel_t* img_gold, my_pixel_t* img_test, int imgHeight, int imgWidth, int imgDepth,
               string img_gold_name, string img_test_name){

    for(int y=0; y<imgHeight; y++){
        for(int x=0; x<imgWidth; x++){
            for(int d=0; d<imgDepth; d++){
                my_pixel_t gold_element = img_gold[x*imgDepth + y*imgWidth*imgDepth + d];
                my_pixel_t test_element = img_test[x*imgDepth + y*imgWidth*imgDepth + d];
                if(test_element != gold_element){
                    //e.g. x=1, y=1, d=31, voc5=..., streamHog=...
                    printf("    x=%d, y=%d, d=%d. %s=%d, %s=%d \n", x, y, d, 
                            img_gold_name.c_str(), gold_element, img_test_name.c_str(), test_element);
                }
            }
        }
    }
}

void diff_hogs(float* hog_gold, float* hog_test, int hogHeight, int hogWidth, int hogDepth,
               string hog_gold_name, string hog_test_name){
    float eps_diff = 0.01;

    for(int y=0; y<hogHeight; y++){
        for(int x=0; x<hogWidth; x++){
            for(int d=0; d<hogDepth; d++){
                float gold_element = hog_gold[x*hogDepth + y*hogWidth*hogDepth + d];
                float test_element = hog_test[x*hogDepth + y*hogWidth*hogDepth + d];
                if( fabs(test_element - gold_element) > eps_diff){
                    //e.g. x=1, y=1, d=31, voc5=..., streamHog=...
                    printf("    x=%d, y=%d, d=%d. %s=%f, %s=%f \n", x, y, d, 
                            hog_gold_name.c_str(), gold_element, hog_test_name.c_str(), test_element);
                }
            }
        }
    }
}

//correctness check
void test_gradients_voc5_vs_streamHOG(){
    streamHog sHog; //streamHog constructor initializes lookup tables & constants (mostly for orientation bins)
    int sbin = 4;

    SimpleImg<uint8_t>img("../../images_640x480/carsgraz_001.image.jpg");

//TODO: use STRIDE instead of WIDTH for the following:
    SimpleImg<uint8_t> ori_stream(img.height, img.width, 1); //out img has just 1 channel
    SimpleImg<uint8_t> ori_voc5(img.height, img.width, 1);
    SimpleImg<int16_t> mag_stream(img.height, img.width, 1); //out img has just 1 channel
    SimpleImg<int16_t> mag_voc5(img.height, img.width, 1); 

  //[mag, ori] = gradient_stream(img)
    sHog.gradient_voc5_reference(img.height, img.width, img.stride, img.n_channels, ori_voc5.n_channels, img.data, ori_voc5.data, mag_voc5.data);
    sHog.gradient_stream(img.height, img.width, img.stride, img.n_channels, ori_stream.n_channels, img.data, ori_stream.data, mag_stream.data); 

    mag_voc5.simple_imwrite("mag_voc5.jpg");
    mag_stream.simple_imwrite("mag_stream.jpg");
    //ori_voc5.simple_imwrite("ori_voc5.jpg");
    //ori_stream.simple_imwrite("ori_stream.jpg");
    ori_voc5.simple_csvwrite("ori_voc5.csv");
    ori_stream.simple_csvwrite("ori_stream.csv");

    #if 1
    printf("diff ori:\n");
    diff_imgs<uint8_t>(ori_voc5.data, ori_stream.data, img.height, img.width, 1, "ori_voc5", "ori_streamHog");
    printf("diff mag:\n");
    diff_imgs<int16_t>(mag_voc5.data, mag_stream.data, img.height, img.width, 1, "mag_voc5", "mag_streamHog");
    #endif
}

//correctness check
void test_computeCells_voc5_vs_streamHOG(){
    streamHog sHog; //streamHog constructor initializes lookup tables & constants (mostly for orientation bins)
    int sbin = 4;

    //SimpleImg<uint8_t> img("./carsgraz001_goofySize_539x471.jpg");
    //SimpleImg<uint8_t> img("./carsgraz001_goofySize_641x480.jpg");
    SimpleImg<uint8_t>img("../../images_640x480/carsgraz_001.image.jpg");

//TODO: use STRIDE instead of WIDTH for the following:
    SimpleImg<uint8_t> ori_stream(img.height, img.width, 1); //out img has just 1 channel
    SimpleImg<uint8_t> ori_voc5(img.height, img.width, 1);
    SimpleImg<int16_t> mag_stream(img.height, img.width, 1); //out img has just 1 channel
    SimpleImg<int16_t> mag_voc5(img.height, img.width, 1); 
    int hogWidth, hogHeight;
    float* hogBuffer_voc5 = allocate_hist(img.height, img.width, sbin,
                                          hogHeight, hogWidth); //hog{Height,Width} are passed by ref.
    float* hogBuffer_streamHog = allocate_hist(img.height, img.width, sbin,
                                               hogHeight, hogWidth); //hog{Height,Width} are passed by ref.
    int hogStride = hogWidth; //TODO: change this?
    float* normImg = (float*)malloc_aligned(32, hogWidth * hogHeight * sizeof(float));

  //[mag, ori] = gradient_stream(img)
    sHog.gradient_voc5_reference(img.height, img.width, img.stride, img.n_channels, ori_voc5.n_channels, img.data, ori_voc5.data, mag_voc5.data);
    sHog.gradient_stream(img.height, img.width, img.stride, img.n_channels, ori_stream.n_channels, img.data, ori_stream.data, mag_stream.data); 

  //hist = computeCells(mag, ori, sbin)
    sHog.computeCells_voc5_reference(img.height, img.width, img.stride, sbin,
                                     ori_stream.data, mag_stream.data, 
                                     hogHeight, hogWidth, hogBuffer_voc5); 
    //sHog.computeCells_stream(img.height, img.width, img.stride, sbin,
    //                         ori_stream.data, mag_stream.data,
    //                         hogHeight, hogWidth, hogBuffer_streamHog);
    //sHog.computeCells_stream_noBoundsCheck(img.height, img.width, img.stride, sbin,
    //                         ori_stream.data, mag_stream.data,
    //                         hogHeight, hogWidth, hogBuffer_streamHog);
    sHog.computeCells_stream_gather(img.height, img.width, img.stride, sbin,
                             ori_stream.data, mag_stream.data,
                             hogHeight, hogWidth, hogBuffer_streamHog);
    int hogDepth = 32;
    diff_hogs(hogBuffer_voc5, hogBuffer_streamHog, hogHeight, hogWidth, hogDepth, "voc5_cells", "streamHog_cells");

  //normImg(x,y) = sum( hist(x,y,0:17) )
    sHog.hogCell_gradientEnergy(hogBuffer_voc5, hogHeight, hogWidth, normImg); //populates normImg

    float* hogBuffer_streamHog_blocks = allocate_hist(img.height, img.width, sbin,
                                                      hogHeight, hogWidth); //will contain final output
}

///correctness check
void test_normalizeCells_voc5_vs_streamHOG(){
    streamHog sHog; //streamHog constructor initializes lookup tables & constants (mostly for orientation bins)
    int sbin = 4;
    int hogDepth = 32;

    SimpleImg<uint8_t>img("../../images_640x480/carsgraz_001.image.jpg");

//TODO: use STRIDE instead of WIDTH for the following:
    SimpleImg<uint8_t> ori_voc5(img.height, img.width, 1);
    SimpleImg<int16_t> mag_voc5(img.height, img.width, 1); 
    int hogWidth, hogHeight;
    float* hogBuffer_voc5 = allocate_hist(img.height, img.width, sbin, hogHeight, hogWidth); //hog{Height,Width} are passed by ref.
    int hogStride = hogWidth; //TODO: change this?
    float* normImg = (float*)malloc_aligned(32, hogWidth * hogHeight * sizeof(float));

  //[mag, ori] = gradient_stream(img)
    sHog.gradient_voc5_reference(img.height, img.width, img.stride, img.n_channels, ori_voc5.n_channels, img.data, ori_voc5.data, mag_voc5.data);

  //hist = computeCells(mag, ori, sbin)
    sHog.computeCells_voc5_reference(img.height, img.width, img.stride, sbin,
                                     ori_voc5.data, mag_voc5.data, 
                                     hogHeight, hogWidth, hogBuffer_voc5); 

    float* hogBuffer_blocks_voc5 = allocate_hist(img.height, img.width, sbin, hogHeight, hogWidth); //will contain final output
    float* hogBuffer_blocks_stream = allocate_hist(img.height, img.width, sbin, hogHeight, hogWidth); //will contain final output

  //blocks = normalizeCells(hist, normImg)
    sHog.hogCell_gradientEnergy(hogBuffer_voc5, hogHeight, hogWidth, normImg); //normImg(x,y) = sum( hist(x,y,0:17) )
    sHog.normalizeCells_voc5(hogBuffer_voc5, normImg, hogBuffer_blocks_voc5, hogHeight, hogWidth); //gold impl

    memset(normImg, 0, hogHeight * hogWidth * sizeof(float)); //normalizeCells_stream repopulates normImg
    sHog.normalizeCells_stream(hogBuffer_voc5, normImg, hogBuffer_blocks_stream, hogHeight, hogWidth); //impl to test

    diff_hogs(hogBuffer_blocks_voc5, hogBuffer_blocks_stream, hogHeight, hogWidth, hogDepth, "voc5_blocks", "streamHog_blocks");
}
// MAIN TEST OF FUNCTIONALITY
// TODO: return a HOG object
// TODO: have a 'bool doWrite' input param
// TODO: have a 'string outFname' param
void test_streamHog_oneScale_manyIter(SimpleImg<uint8_t> &img, int sbin, streamHog sHog){

    int n_iter = 10000; //not really "iterating" -- just number of times to run the experiment
    if(n_iter < 10000){
        printf("WARNING: n_iter = %d. For statistical significance, we recommend n_iter=10000 or greater. \n", n_iter);
    }
    //SimpleImg img(height, width, n_channels);

    //SimpleImg<uint8_t> img("../../images_640x480/carsgraz_001.image.jpg");
    SimpleImg<uint8_t> ori(img.height, img.stride, 1); //out img has just 1 channel
    SimpleImg<int16_t> mag(img.height, img.stride, 1); //out img has just 1 channel
    int hogWidth, hogHeight;
    float* hogBuffer = allocate_hist(img.height, img.width, sbin,
                                     hogHeight, hogWidth); //hog{Height,Width} are passed by ref.
    float* hogBuffer_blocks = allocate_hist(img.height, img.width, sbin,
                                            hogHeight, hogWidth); //for normalized result
    float* normImg = (float*)malloc_aligned(32, hogWidth * hogHeight * sizeof(float));

    //16bit is temporary -- for speed test.
    int16_t* hogBuffer_16bit = allocate_hist_16bit(img.height, img.width, sbin,
                                     hogHeight, hogWidth); //hog{Height,Width} are passed by ref.

  //[mag, ori] = gradient_stream(img)
    double start_timer = read_timer();
    for(int i=0; i<n_iter; i++){
        sHog.gradient_stream(img.height, img.width, img.stride, img.n_channels, ori.n_channels, img.data, ori.data, mag.data); 
        //sHog.gradient_voc5_reference(img.height, img.width, img.stride, img.n_channels, ori.n_channels, img.data, ori.data, mag.data);
    }

    double stream_time = (read_timer() - start_timer) / n_iter;
    double gb_to_copy = img.width * img.height * img.n_channels * sizeof(uint8_t) / 1e9;
    double gb_per_sec = gb_to_copy / (stream_time/1000); //convert stream_time from ms to sec
    printf("avg (mag, ori) stream time = %f ms, %f GB/s \n", stream_time, gb_per_sec);

  //hist = computeCells(mag, ori, sbin)
    start_timer = read_timer();

    for(int i=0; i<n_iter; i++){
        //ori and mag both have size {img.height, img.width}
 
//        sHog.computeCells_voc5_reference(img.height, img.width, img.stride, sbin,
//                                         ori.data, mag.data, 
//                                         hogHeight, hogWidth, hogBuffer); 

//        sHog.computeCells_stream(img.height, img.width, img.stride, sbin,
//                                 ori.data, mag.data,
//                                 hogHeight, hogWidth, hogBuffer);

//        sHog.computeCells_stream_output_16bit(img.height, img.width, img.stride, sbin,
//                                 ori.data, mag.data,
//                                 hogHeight, hogWidth, hogBuffer_16bit);

//        sHog.computeCells_stream_noBoundsCheck(img.height, img.width, img.stride, sbin,
//                                 ori.data, mag.data,
//                                 hogHeight, hogWidth, hogBuffer);
                                

        sHog.computeCells_stream_gather(img.height, img.width, img.stride, sbin,
                                 ori.data, mag.data,
                                 hogHeight, hogWidth, hogBuffer);

    }

    stream_time = (read_timer() - start_timer) / n_iter;
    int hogDepth = 32;
    gb_to_copy = hogWidth * hogHeight * hogDepth * sizeof(float) / 1e9; //TODO: change 'float' to new data type if necessary
    gb_per_sec = gb_to_copy / (stream_time/1000); //convert stream_time from ms to sec
    printf("avg hogCell stream time = %f ms, %f GB/s \n", stream_time, gb_per_sec);

  //normImg(x,y) = sum( hist(x,y,0:17) )
    start_timer = read_timer();

    for(int i=0; i<n_iter; i++){
        sHog.hogCell_gradientEnergy(hogBuffer, hogHeight, hogWidth, normImg); //populates normImg
    }

    stream_time = (read_timer() - start_timer) / n_iter;
    gb_to_copy = hogWidth * hogHeight * 18 * sizeof(float) / 1e9; //for each hog cell: read 18 elements, write 1
    gb_per_sec = gb_to_copy / (stream_time/1000); //convert stream_time from ms to sec
    printf("avg hogCell_gradientEnergy stream time = %f ms, %f GB/s \n", stream_time, gb_per_sec); 

  //blocks = normalizeCells(hist, normImg)
    start_timer = read_timer();

    for(int i=0; i<n_iter; i++){
        sHog.normalizeCells_voc5(hogBuffer, normImg, hogBuffer_blocks,
                                hogHeight, hogWidth);
    }

    stream_time = (read_timer() - start_timer) / n_iter;
    gb_to_copy = hogWidth * hogHeight * 18 * sizeof(float) / 1e9; //TODO: think about amt of data to stream
    gb_per_sec = gb_to_copy / (stream_time/1000); //convert stream_time from ms to sec
    printf("avg normalizeCells_voc5 stream time = %f ms, %f GB/s \n", stream_time, gb_per_sec);

    free(hogBuffer);
    free(hogBuffer_blocks);
}

//wrapper that does a basic sanity check of SPEED
void test_streamHog_oneScale_default(){
    int sbin = 4; 
    streamHog sHog; //streamHog constructor initializes lookup tables & constants (mostly for orientation bins)
    SimpleImg<uint8_t> img("../../images_640x480/carsgraz_001.image.jpg");
    test_streamHog_oneScale_manyIter(img, sbin, sHog); //DEBUG segfault: if I remove this, then we don't get a segfault on ~img{ free(img.data) }
}

//no timers in here ... just crank through it once
void test_streamHog_oneScale_oneIter(SimpleImg<uint8_t> &img, int sbin, streamHog sHog){

    SimpleImg<uint8_t> ori(img.height, img.stride, 1); //out img has just 1 channel
    SimpleImg<int16_t> mag(img.height, img.stride, 1); //out img has just 1 channel
    int hogWidth, hogHeight;
    
     
    float* hogBuffer = allocate_hist(img.height, img.width, sbin,
                                     hogHeight, hogWidth); //hog{Height,Width} are passed by ref.
    float* hogBuffer_blocks = allocate_hist(img.height, img.width, sbin,
                                            hogHeight, hogWidth); //for normalized result
    float* normImg = (float*)malloc_aligned(32, hogWidth * hogHeight * sizeof(float));

  //[mag, ori] = gradient_stream(img)
    sHog.gradient_stream(img.height, img.width, img.stride, img.n_channels, ori.n_channels, img.data, ori.data, mag.data); 
    //sHog.gradient_voc5_reference(img.height, img.width, img.stride, img.n_channels, ori.n_channels, img.data, ori.data, mag.data);

    //ori and mag both have size {img.height, img.width}

//    sHog.computeCells_voc5_reference(img.height, img.width, img.stride, sbin,
//                                      ori.data, mag.data, 
//                                      hogHeight, hogWidth, hogBuffer); 

//    sHog.computeCells_stream(img.height, img.width, img.stride, sbin,
//                             ori.data, mag.data,
//                             hogHeight, hogWidth, hogBuffer);

//    sHog.computeCells_stream_noBoundsCheck(img.height, img.width, img.stride, sbin,
//                             ori.data, mag.data,
//                             hogHeight, hogWidth, hogBuffer);

    sHog.computeCells_stream_gather(img.height, img.width, img.stride, sbin,
                             ori.data, mag.data,
                             hogHeight, hogWidth, hogBuffer);

  //normImg(x,y) = sum( hist(x,y,0:17) )
    sHog.hogCell_gradientEnergy(hogBuffer, hogHeight, hogWidth, normImg); //populates normImg

  //blocks = normalizeCells(hist, normImg)
    sHog.normalizeCells_voc5(hogBuffer, normImg, hogBuffer_blocks,
                             hogHeight, hogWidth);

    free(hogBuffer);
    free(hogBuffer_blocks);
}

//hand-coded impl of pyramid. (will modularize it better eventually)
void test_streamHog_pyramid(){
    printf("test_streamHog_pyramid() \n");
    int nLevels = 30; //TODO: compute this based on img size
    int interval = 10;
    int n_iter = 100; //not really "iterating" -- just number of times to run the experiment
    if(n_iter < 10){
        printf("    WARNING: n_iter = %d. For statistical significance, we recommend n_iter=10 or greater. \n", n_iter);
    }

    streamHog sHog; //streamHog constructor initializes lookup tables & constants (mostly for orientation bins)
    float sc = pow(2, 1 / (float)interval);
    cv::Mat img_Mat = cv::imread("../../images_640x480/carsgraz_001.image.jpg"); 

    vector< SimpleImg<uint8_t>* > imgPyramid(nLevels); //TODO: have these *not* be pointers?


    //TODO: create an imgPyramid function.
    #pragma omp parallel for
    for(int i=0; i<interval; i++){
        float downsampleFactor = 1/pow(sc, i);
        //printf("downsampleFactor = %f \n", downsampleFactor);

        cv::Mat img_scaled = downsampleWithOpenCV(img_Mat, downsampleFactor);
        imgPyramid[i] = new SimpleImg<uint8_t>(img_scaled);
        img_scaled = downsampleWithOpenCV(img_Mat, downsampleFactor/2);
        imgPyramid[i + interval] = new SimpleImg<uint8_t>(img_scaled);
    }

    double start_time = read_timer();

    for(int iter=0; iter<n_iter; iter++){ //do several runs, take the avg time
#pragma omp parallel for //gives 'bus error' even if loop is basically empty.
        for(int i=0; i<interval; i++){
            //double perScale_start = read_timer();

            //TODO: catch output HOGs from each scale
            #if 0 
            test_streamHog_oneScale_manyIter(*imgPyramid[i], 4, sHog); //sbin=4 -- hogPyramid[i]
            test_streamHog_oneScale_manyIter(*imgPyramid[i+interval], 8, sHog); //sbin=8 -- hogPyramid[i + interval]
            test_streamHog_oneScale_manyIter(*imgPyramid[i + 2*interval], 8, sHog); //sbin=8 -- hogPyramid[i + 2*interval]
            #endif

            //TODO: avoid recomputing (ori, mag) for both sbin=4 and sbin=8!!!
            #if 1 
            test_streamHog_oneScale_oneIter(*imgPyramid[i], 4, sHog); //sbin=4 -- hogPyramid[i]
            test_streamHog_oneScale_oneIter(*imgPyramid[i], 8, sHog); //sbin=8 -- hogPyramid[i + interval]
            test_streamHog_oneScale_oneIter(*imgPyramid[i + interval], 8, sHog); //sbin=8 -- hogPyramid[i + 2*interval]
            #endif

            //double perScale_end = read_timer() - perScale_start;
            //printf("    scale %d, hog extraction time = %f ms \n", i, perScale_end);
        }    
    }

    double end_timer = read_timer() - start_time;
    printf("avg time for multiscale = %f ms \n", end_timer/n_iter);

    for(int i=0; i<nLevels; i++){
        delete imgPyramid[i];
    }
}


