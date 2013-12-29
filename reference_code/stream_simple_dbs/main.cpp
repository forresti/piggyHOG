#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cassert>
#include <xmmintrin.h>
#include <pmmintrin.h> //for _mm_hadd_pd()

#include "ForrestImg.h"
#include "helpers.h"
using namespace std;
//using namespace cv;

//note: ForrestImg's pixel_t is defined in ForrestImg.h
void stream_simple(int height, 
		   int width, 
		   int stride,
		   pixel_t *__restrict__ img, 
		   pixel_t *__restrict__ outImg)
{
    for(int y=0; y<height; y++){
        for(int x=0; x<width; x++){
            outImg[y *stride + x] = img[y*stride + x];
        }
    }
}

void stream_sse(int height,
           int width,
           int stride,
           pixel_t *__restrict__ img,
           pixel_t *__restrict__ outImg){

    //loadSize example: (128 bits = 16 bytes) / (4 bytes per float) = 4 bytes
    int loadSize = sizeof(__m128) / sizeof(pixel_t);

    for(int y=0; y<height; y++){
        //for(int x=0; x<width; x++){
        for(int x=0; x<width; x+=loadSize){

//TODO: use pixel_t=float for this experiment
#if 0
            __m128 in_vec = _mm_load_ps(&img[y*stride + x]); //floating-pt version doesn't like cast to __m128i. shrug.

            __m128 in_vec_sqrt = _mm_sqrt_ps(in_vec); //test
            _mm_store_ps(&outImg[y*stride + x], in_vec_sqrt);
#endif
        }
    }
}

//try 8bit vs 'expand to 16bit' addition
void stream_sse_addition(int height, int width, int stride,
                         pixel_t *__restrict__ img, pixel_t *__restrict__ outImg){


    //loadSize example: (128 bits = 16 bytes) / (4 bytes per float) = 4 bytes
    int loadSize = sizeof(__m128) / sizeof(pixel_t);
    for(int y=0; y<height; y++){
        for(int x=0; x<width; x+=loadSize){

//TODO: use pixel_t=uchar for this experiment
#if 1
            __m128i in_vec = _mm_load_si128((__m128i*)(&img[y*stride + x])); //floating-pt version doesn't like cast to __m128i. shrug.
            
            #if 1 //8-bit add
            __m128i out_8bit = _mm_add_epi8(in_vec, in_vec); 
            #endif

            #if 0 //16bit add
            __m128i in_vec_0 = _mm_unpacklo_epi8(in_vec, _mm_setzero_si128());
            __m128i in_vec_1 = _mm_unpackhi_epi8(in_vec, _mm_setzero_si128());

            __m128i out_16bit_0 = _mm_add_epi16(in_vec_0, in_vec_0);
            __m128i out_16bit_1 = _mm_add_epi16(in_vec_1, in_vec_1);

            __m128i out_8bit = _mm_packs_epi16(out_16bit_0, out_16bit_1);
            #endif

            _mm_store_si128((__m128i*)(&outImg[y*stride + x]), out_8bit);
#endif
        }
    }
}

void fill_img_with_garbage(ForrestImg& img){
    for(int y=0; y<img.height; y++){
        for(int x=0; x<img.width; x++){
            img.data[y*img.stride + x] = (pixel_t)((y*img.stride + x)%256); //junk data
        }
    }
}

int main (int argc, char **argv)
{
    int ALIGN_IN_BYTES = 256;

    int height = 480;
    int width = 640*3;
    int stride = width + (ALIGN_IN_BYTES - width%ALIGN_IN_BYTES); //thanks: http://stackoverflow.com/questions/2403631
    //int stride = width;

printf("stride = %d \n", stride);

    ForrestImg img(height, width, stride);
    ForrestImg outImg(height, width, stride);
    fill_img_with_garbage(img);
    int n_iter = 1000;

    double start_timer = read_timer();
    for(int i=0; i<n_iter; i++){
        stream_simple(img.height, img.width, img.stride, img.data, outImg.data);
        //stream_sse(img.height, img.width, img.stride, img.data, outImg.data);
        stream_sse_addition(img.height, img.width, img.stride, img.data, outImg.data); //test 8bit vs unpack 16bit addition
        //memcpy(outImg.data, img.data, height * stride * sizeof(pixel_t));
    }
    double stream_time = (read_timer() - start_timer) / n_iter;
    double gb_to_copy = width * height * sizeof(pixel_t) / 1e9;
    double gb_per_sec = gb_to_copy / (stream_time/1000); //convert stream_time from ms to sec
    printf("avg stream time = %f ms, %f GB/s \n", stream_time, gb_per_sec);
    

    return 0;
}


