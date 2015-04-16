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
#include "driver_streamHog.h"
using namespace std;


//TODO function:
//vector< SimpleImg<uint8_t>* > get_imgPyramid(...args...)

//HACK: assumes we're using the protocol "sbin=4 for top octave; sbin=8 otherwise"
int get_sbin_for_scale(int scaleIdx, int interval){
    int sbin;
    if(scaleIdx < interval) sbin=4; //top octave
    else sbin = 8; //other octaves
    return sbin;
}

//hand-coded impl of pyramid. (will modularize it better eventually)
// typical: img pyramid:16.901926 ms, gradients: 4.586978 ms, hist: 9.008650 ms, norm: 6.529126 ms
void streamHog_pyramid(){
    printf("streamHog_pyramid() \n");
    int nLevels = 40; //TODO: compute this based on img size
    int interval = 10;
    int n_iter = 100; //not really "iterating" -- just number of times to run the experiment
    if(n_iter < 10){
        printf("    WARNING: n_iter = %d. For statistical significance, we recommend n_iter=10 or greater. \n", n_iter);
    }

    streamHog sHog; //streamHog constructor initializes lookup tables & constants (mostly for orientation bins)
    float sc = pow(2, 1 / (float)interval);
    cv::Mat img_Mat = cv::imread("../../images_640x480/carsgraz_001.image.jpg"); 

    vector< SimpleImg<uint8_t>* > imgPyramid(nLevels);
    vector< SimpleImg<uint8_t>* > ori(nLevels); //(img.height, img.stride, 1); //out img has just 1 channel
    vector< SimpleImg<int16_t>* > mag(nLevels); //(img.height, img.stride, 1); //out img has just 1 channel
    vector< float* > hogBuffer(nLevels);
    vector< float* > normImg(nLevels);
    vector< float* > hogBuffer_blocks(nLevels); //this is the output HOG pyramid
    vector< int > hogHeight(nLevels);
    vector< int > hogWidth(nLevels);

    double start_time = read_timer();
    double img_pyra_time;
    double grad_time;
    double hist_time;
    double norm_time;

    for(int iter=0; iter<n_iter; iter++){ //do several runs, take the avg time

        //TODO: try sbin=4 and smaller resolution for full pyramid (evaluate accuracy in voc-release5)

//step 1: image pyramid
        double img_pyra_start = read_timer();

        //TODO -- make a 'vector<SimpleImg*> = createImgPyramid()' function
        assert( nLevels == 4*interval ); //TODO: relax this.
        #pragma omp parallel for
        for(int i=0; i<interval; i++){
            float downsampleFactor = 1/pow(sc, i);
            //printf("downsampleFactor = %f \n", downsampleFactor);

            //top 10 scales
            cv::Mat img_scaled = downsampleWithOpenCV(img_Mat, downsampleFactor);
            imgPyramid[i] = new SimpleImg<uint8_t>(img_scaled); //use w/ sbin=4

            imgPyramid[i + interval] = new SimpleImg<uint8_t>(img_scaled); //use w/ sbin=8

            //next 10 scales
            img_scaled = downsampleWithOpenCV(img_Mat, downsampleFactor/2);
            imgPyramid[i + 2*interval] = new SimpleImg<uint8_t>(img_scaled); //use w/ sbin=8

            //bottom 10 scales
            img_scaled = downsampleWithOpenCV(img_Mat, downsampleFactor/4);
            imgPyramid[i + 3*interval] = new SimpleImg<uint8_t>(img_scaled); //use w/ sbin=8
        }

//step 1.1: now that we know img dimensions, allocate memory for HOG stuff

        #pragma omp parallel for
        for(int s=0; s<nLevels; s++){

            int sbin = get_sbin_for_scale(s, interval);

            ori[s] = new SimpleImg<uint8_t>(imgPyramid[s]->height, imgPyramid[s]->stride, 1);
            mag[s] = new SimpleImg<int16_t>(imgPyramid[s]->height, imgPyramid[s]->stride, 1);

            hogBuffer[s] = allocate_hist(imgPyramid[s]->height, imgPyramid[s]->width, sbin,
                                             hogHeight[s], hogWidth[s]); //hog{Height,Width} are passed by ref.
            hogBuffer_blocks[s] = allocate_hist(imgPyramid[s]->height, imgPyramid[s]->width, sbin,
                                                    hogHeight[s], hogWidth[s]); //for normalized result
            normImg[s] = (float*)malloc_aligned(32, hogWidth[s] * hogHeight[s] * sizeof(float));

        }

        img_pyra_time += (read_timer() - img_pyra_start); 



//step 2: gradients
        double grad_start = read_timer();

        #pragma omp parallel for
        for(int s=0; s<nLevels; s++){

            int sbin = get_sbin_for_scale(s, interval);

            //[mag, ori] = gradient_stream(img)
            sHog.gradient_stream(imgPyramid[s]->height, imgPyramid[s]->width, imgPyramid[s]->stride, 
                                 imgPyramid[s]->n_channels, ori[s]->n_channels, imgPyramid[s]->data, ori[s]->data, mag[s]->data);
        }

        grad_time += (read_timer() - grad_start);

//step 3: histogram cells
        double hist_start = read_timer();

        #pragma omp parallel for
        for(int s=0; s<nLevels; s++){

            int sbin = get_sbin_for_scale(s, interval);
 
            sHog.computeCells_stream_gather(imgPyramid[s]->height, imgPyramid[s]->width, imgPyramid[s]->stride, sbin,
                                           ori[s]->data, mag[s]->data, hogHeight[s], hogWidth[s], hogBuffer[s]);
        }

        hist_time += (read_timer() - hist_start);

//step 4: normalize cells into blocks
        double norm_start = read_timer();

        #pragma omp parallel for
        for(int s=0; s<nLevels; s++){

            //normImg(x,y) = sum( hist(x,y,0:17) )
            sHog.hogCell_gradientEnergy(hogBuffer[s], hogHeight[s], hogWidth[s], normImg[s]); //populates normImg

            //blocks = normalizeCells(hist, normImg)
            sHog.normalizeCells_voc5(hogBuffer[s], normImg[s], hogBuffer_blocks[s],
                                     hogHeight[s], hogWidth[s]);
        }

        norm_time += (read_timer() - norm_start);
    
        for(int s=0; s<nLevels; s++){
            delete imgPyramid[s];
            delete ori[s];
            delete mag[s];
            free(hogBuffer[s]);
            free(hogBuffer_blocks[s]);
            free(normImg[s]);
        }
    }
    double end_timer = read_timer() - start_time;
    printf("avg time for multiscale = %f ms \n", end_timer/n_iter);
    printf("img pyramid:%f ms, gradients: %f ms, hist: %f ms, norm: %f ms\n", img_pyra_time/n_iter, grad_time/n_iter, hist_time/n_iter, norm_time/n_iter);


}


//TEMP: version w/ outer omp loop instead of having many small omp loops.
// This is a bit faster. typical: img pyramid:16.244524 ms, grad+hist+norm: 15.750964 ms
void outerLoopParallel_streamHog_pyramid(){
    printf("outerLoopParallel_streamHog_pyramid() \n");
    int nLevels = 40; //TODO: compute this based on img size
    int interval = 10;
    int n_iter = 100; //not really "iterating" -- just number of times to run the experiment
    if(n_iter < 10){
        printf("    WARNING: n_iter = %d. For statistical significance, we recommend n_iter=10 or greater. \n", n_iter);
    }

    streamHog sHog; //streamHog constructor initializes lookup tables & constants (mostly for orientation bins)
    float sc = pow(2, 1 / (float)interval);
    cv::Mat img_Mat = cv::imread("../../images_640x480/carsgraz_001.image.jpg"); 

    vector< SimpleImg<uint8_t>* > imgPyramid(nLevels);
    vector< SimpleImg<uint8_t>* > ori(nLevels); //(img.height, img.stride, 1); //out img has just 1 channel
    vector< SimpleImg<int16_t>* > mag(nLevels); //(img.height, img.stride, 1); //out img has just 1 channel
    vector< float* > hogBuffer(nLevels);
    vector< float* > normImg(nLevels);
    vector< float* > hogBuffer_blocks(nLevels); //this is the output HOG pyramid
    vector< int > hogHeight(nLevels);
    vector< int > hogWidth(nLevels);

    double start_time = read_timer();
    double img_pyra_time;
    double grad_time;
    double hist_time;
    double norm_time;
    double grad_hist_norm_time = 0;

    for(int iter=0; iter<n_iter; iter++){ //do several runs, take the avg time

        //TODO: try sbin=4 and smaller resolution for full pyramid (evaluate accuracy in voc-release5)

//step 1: image pyramid
        double img_pyra_start = read_timer();

        //TODO -- make a 'vector<SimpleImg*> = createImgPyramid()' function
        assert( nLevels == 4*interval ); //TODO: relax this.
        #pragma omp parallel for
        for(int i=0; i<interval; i++){
            float downsampleFactor = 1/pow(sc, i);
            //printf("downsampleFactor = %f \n", downsampleFactor);

            //top 10 scales
            cv::Mat img_scaled = downsampleWithOpenCV(img_Mat, downsampleFactor);
            imgPyramid[i] = new SimpleImg<uint8_t>(img_scaled); //use w/ sbin=4

            imgPyramid[i + interval] = new SimpleImg<uint8_t>(img_scaled); //use w/ sbin=8

            //next 10 scales
            img_scaled = downsampleWithOpenCV(img_Mat, downsampleFactor/2);
            imgPyramid[i + 2*interval] = new SimpleImg<uint8_t>(img_scaled); //use w/ sbin=8

            //bottom 10 scales
            img_scaled = downsampleWithOpenCV(img_Mat, downsampleFactor/4);
            imgPyramid[i + 3*interval] = new SimpleImg<uint8_t>(img_scaled); //use w/ sbin=8
        }

//step 1.1: now that we know img dimensions, allocate memory for HOG stuff

        #pragma omp parallel for
        for(int s=0; s<nLevels; s++){

            int sbin = get_sbin_for_scale(s, interval);

            ori[s] = new SimpleImg<uint8_t>(imgPyramid[s]->height, imgPyramid[s]->stride, 1);
            mag[s] = new SimpleImg<int16_t>(imgPyramid[s]->height, imgPyramid[s]->stride, 1);

            hogBuffer[s] = allocate_hist(imgPyramid[s]->height, imgPyramid[s]->width, sbin,
                                             hogHeight[s], hogWidth[s]); //hog{Height,Width} are passed by ref.
            hogBuffer_blocks[s] = allocate_hist(imgPyramid[s]->height, imgPyramid[s]->width, sbin,
                                                    hogHeight[s], hogWidth[s]); //for normalized result
            normImg[s] = (float*)malloc_aligned(32, hogWidth[s] * hogHeight[s] * sizeof(float));
        }

        img_pyra_time += (read_timer() - img_pyra_start);


        double grad_hist_norm_start = read_timer();
//step 2: gradients
        #pragma omp parallel for
        for(int s=0; s<nLevels; s++){

            int sbin = get_sbin_for_scale(s, interval);

            //[mag, ori] = gradient_stream(img)
            sHog.gradient_stream(imgPyramid[s]->height, imgPyramid[s]->width, imgPyramid[s]->stride, 
                                 imgPyramid[s]->n_channels, ori[s]->n_channels, imgPyramid[s]->data, ori[s]->data, mag[s]->data);

//step 3: histogram cells
            sHog.computeCells_stream_gather(imgPyramid[s]->height, imgPyramid[s]->width, imgPyramid[s]->stride, sbin,
                                           ori[s]->data, mag[s]->data, hogHeight[s], hogWidth[s], hogBuffer[s]);


//step 4: normalize cells into blocks
            //normImg(x,y) = sum( hist(x,y,0:17) )
            sHog.hogCell_gradientEnergy(hogBuffer[s], hogHeight[s], hogWidth[s], normImg[s]); //populates normImg

            //blocks = normalizeCells(hist, normImg)
            sHog.normalizeCells_voc5(hogBuffer[s], normImg[s], hogBuffer_blocks[s],
                                     hogHeight[s], hogWidth[s]);
   
        }
        grad_hist_norm_time += (read_timer() - grad_hist_norm_start);
 
        for(int s=0; s<nLevels; s++){
            delete imgPyramid[s];
            delete ori[s];
            delete mag[s];
            free(hogBuffer[s]);
            free(hogBuffer_blocks[s]);
            free(normImg[s]);
        }
    }
    double end_timer = read_timer() - start_time;
    printf("avg time for multiscale = %f ms \n", end_timer/n_iter);
    //printf("img pyramid:%f ms, gradients: %f ms, hist: %f ms, norm: %f ms\n", img_pyra_time/n_iter, grad_time/n_iter, hist_time/n_iter, norm_time/n_iter);
    printf("img pyramid:%f ms, grad+hist+norm: %f ms\n", img_pyra_time/n_iter, grad_hist_norm_time/n_iter);

}



