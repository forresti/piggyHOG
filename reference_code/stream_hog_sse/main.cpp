#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cassert>
#include <xmmintrin.h>
#include <pmmintrin.h> //for _mm_hadd_pd()

#include "SimpleImg.h"
#include "helpers.h"
using namespace std;
char ATAN2_TABLE[512][512]; //signed char (values are -18 to 18)

__m128i max_channel_ori(__m128i ori_ch[3], __m128i mag_ch[3]){
    __m128i ori_max;
    //stub
    return ori_max;
}

//enables us to load 8-bit values, but work in 16-bit. 
//      TODO: struct for the inputs / outputs?
void upcast_8bit_to_16bit(__m128i in_xLo,     __m128i in_xHi,     __m128i in_yLo,     __m128i in_yHi,
                          __m128i &out_xLo_0, __m128i &out_xHi_0, __m128i &out_yLo_0, __m128i &out_yHi_0, //bottom bits, in 16-bit
                          __m128i &out_xLo_1, __m128i &out_xHi_1, __m128i &out_yLo_1, __m128i &out_yHi_1) //top bits,    in 16-bit
{
    //convert inputs for gradY to 16 bits
    out_xLo_0 = _mm_unpacklo_epi8(in_xLo, _mm_setzero_si128()); //unsigned cast to 16-bit ints -- bottom bits 
    out_xHi_0 = _mm_unpacklo_epi8(in_xHi, _mm_setzero_si128()); 
    out_xLo_1 = _mm_unpackhi_epi8(in_xLo, _mm_setzero_si128()); //unsigned cast to 16-bit ints -- top bits
    out_xHi_1 = _mm_unpackhi_epi8(in_xHi, _mm_setzero_si128());

    //convert inputs for gradY to 16 bits
    out_yLo_0 = _mm_unpacklo_epi8(in_yLo, _mm_setzero_si128()); //unsigned cast to 16-bit ints -- bottom bits 
    out_yHi_0 = _mm_unpacklo_epi8(in_yHi, _mm_setzero_si128()); 
    out_yLo_1 = _mm_unpackhi_epi8(in_yLo, _mm_setzero_si128()); //unsigned cast to 16-bit ints -- top bits
    out_yHi_1 = _mm_unpackhi_epi8(in_yHi, _mm_setzero_si128());

    //not sure if this is useful or not: -- something to do with sign extension
    //out_yLo_0 = _mm_cvtepu8_epi16(out_yLo_0); //'officially' convert the '8-bit with room for 16-bit' to 16-bit?
    //out_yLo_1 = _mm_cvtepu8_epi16(out_yLo_1);
    //out_yHi_0 = _mm_cvtepu8_epi16(out_yHi_0);
    //out_yHi_1 = _mm_cvtepu8_epi16(out_yHi_1);
}
void gradient_sse(int height, int width, int stride, int n_channels_input, int n_channels_output,
                            pixel_t *__restrict__ img, pixel_t *__restrict__ outOri, pixel_t *__restrict__ outMag){
    assert(n_channels_input == 3);
    assert(n_channels_output == 1);
    assert(sizeof(__m128i) == 16);
    int loadSize = sizeof(__m128i); // 16 bytes = 128 bits

    __m128i xLo, xHi, yLo, yHi; //packed 8-bit
    __m128i xLo_0, xHi_0, yLo_0, yHi_0; //bottom bits: upcast from 8-bit to 16-bit
    __m128i xLo_1, xHi_1, yLo_1, yHi_1; //top bits: upcast from 8-bit to 16-bit
    __m128i gradX_ch[3],   gradY_ch[3];   //packed 8-bit
    __m128i gradX_ch_0[3], gradY_ch_0[3]; //bottom bits: upcast from 8-bit to 16-bit
    __m128i gradX_ch_1[3], gradY_ch_1[3]; //top bits: upcast from 8-bit to 16-bit
    __m128i mag_ch_0[3]; //top bits
    __m128i mag_ch_1[3]; //bottom bits

    for(int y=2; y<height-2; y++){
        for(int x=0; x < stride-2; x+=loadSize){ //(stride-2) to avoid falling off the end when doing (location+2) to get xHi

            for(int channel=0; channel<3; channel++){ //TODO: unroll channels
                //xLo = _mm_loadu_pu8(&img[y*stride + x + channel*height*stride - 1]);    //load eight 1-byte unsigned char pixels
                xLo = _mm_loadu_si128( (__m128i*)(&img[y*stride + x + channel*height*stride    ]) ); //load eight 1-byte unsigned char pixels
                xHi = _mm_loadu_si128( (__m128i*)(&img[y*stride + x + channel*height*stride + 2]) ); //index as chars, THEN cast to __m128i*  

                yLo = _mm_load_si128( (__m128i*)(&img[y*stride + x + channel*height*stride           ]) ); //y-dim is a long stride, easier to do aligned loads
                yHi = _mm_load_si128( (__m128i*)(&img[y*stride + x + channel*height*stride + 2*stride]) );
                upcast_8bit_to_16bit(xLo, xHi, yLo, yHi,
                                     xLo_0, xHi_0, yLo_0, yHi_0,
                                     xLo_1, xHi_1, yLo_1, yHi_1);

                //gradX_ch[channel] =  _mm_sub_epi8(xHi, xLo); //overflows ... need 16-bit
                //gradY_ch[channel] =  _mm_sub_epi8(yHi, yLo); //overflows ... need 16-bit

                gradX_ch_0[channel] =  _mm_sub_epi16(xHi_0, xLo_0); // xHi[0:3] - xLo[0:3]
                gradX_ch_1[channel] =  _mm_sub_epi16(xHi_1, xLo_1);  // xHi[4:7] - xLo[4:7]
                gradX_ch[channel] = _mm_packs_epi16(gradX_ch_0[channel], gradX_ch_1[channel]); //16-bit -> 8bit (temporary ... typically, we'd pack up the results later in the pipeline)

                gradY_ch_0[channel] =  _mm_sub_epi16(yHi_0, yLo_0);
                gradY_ch_1[channel] =  _mm_sub_epi16(yHi_1, yLo_1); 
                gradY_ch[channel] = _mm_packs_epi16(gradY_ch_0[channel], gradY_ch_1[channel]); //temporary ... typically, we'd pack up the results later in the pipeline.

                //argh, _mm_mul_epi16 doesn't exist. we only seem to have _mm_mullo_epi16 and _mm_mulhi_epi16
                //mag_ch_0[channel] = _mm_add_epi16( _mm_mul_epi16(gradX_ch_0[channel], gradX_ch_0[channel]), 
                //                                   _mm_mul_epi16(gradX_ch_0[channel], gradX_ch_0[channel]) ); //gradX^2 + gradY^2

                _mm_store_si128( (__m128i*)(&outOri[y*stride + x]), gradX_ch[channel] ); //outOri[y][x : x+7] = gradX_ch[channel] -- just a test, doesnt make much sense
                _mm_store_si128( (__m128i*)(&outMag[y*stride + x]), gradY_ch[channel] ); //aligned stores are easy here...it's all divisible by loadSize.
            }
        }
    }
}


//no attempt at vectorization...just checking correctness.
void gradient_wideload_unvectorized(int height, int width, int stride, int n_channels_input, int n_channels_output,
                            pixel_t *__restrict__ img, pixel_t *__restrict__ outOri, pixel_t *__restrict__ outMag){
    assert(n_channels_input == 3);
    assert(n_channels_output == 1);

    long int xLo[3]; //input data for gradX
    long int xHi[3];
    long int yLo[3]; //input data for gradY
    long int yHi[3];
    int loadSize = 8; //long int

    int mag_ch[3];
    //int16_t mag_ch[3];
    int16_t gradX_ch[3];
    int16_t gradY_ch[3];
    int16_t ori_ch[3];
    
    long int* img_long_int = reinterpret_cast<long int*>(img);

    for(int y=2; y<height-2; y++){
        //for(int x=2; x < width-2; x++){ //replaced with unrolling
        //for(int x_tile=2; x_tile < (stride-loadSize); x_tile+=loadSize){
        for(int x_tile=2; x_tile < (stride/loadSize); x_tile++){
            //TODO: load a 8-byte long int for each grad_ch
            xLo[0] = img_long_int[y*(stride/loadSize) + x_tile +                 - 1];
            xHi[0] = img_long_int[y*(stride/loadSize) + x_tile +                 + 1]; 

            for(int x_inner=0; x_inner < loadSize; x_inner++){
                int x = (x_tile-1)*loadSize + x_inner + 1; //(x_tile-1)...+1 -> because we're starting from x_tile=2 instead of x_tile=0
 
                //gradX_ch[0] = (int16_t)img[y*stride + x +                 + 1] - (int16_t)img[y*stride + x                   - 1];
                gradX_ch[0] = reinterpret_cast<unsigned char*>(xHi)[x_inner] - reinterpret_cast<unsigned char*>(xLo)[x_inner]; //test loading larger words
                gradX_ch[1] = (int16_t)img[y*stride + x + 1*height*stride + 1] - (int16_t)img[y*stride + x + 1*height*stride - 1];
                gradX_ch[2] = (int16_t)img[y*stride + x + 2*height*stride + 1] - (int16_t)img[y*stride + x + 2*height*stride - 1];

                gradY_ch[0] = (int16_t)img[y*stride + x +                 + stride] - (int16_t)img[y*stride + x                   - stride];
                gradY_ch[1] = (int16_t)img[y*stride + x + 1*height*stride + stride] - (int16_t)img[y*stride + x + 1*height*stride - stride];
                gradY_ch[2] = (int16_t)img[y*stride + x + 2*height*stride + stride] - (int16_t)img[y*stride + x + 2*height*stride - stride];

                mag_ch[0] = gradX_ch[0]*gradX_ch[0] + gradY_ch[0]*gradY_ch[0];
                mag_ch[1] = gradX_ch[1]*gradX_ch[1] + gradY_ch[1]*gradY_ch[1];
                mag_ch[2] = gradX_ch[2]*gradX_ch[2] + gradY_ch[2]*gradY_ch[2];

                int16_t gradX, gradY;
                int mag_max = 0;
                int16_t mag_argmax = 0;

                for(int i=0; i<3; i++){
                    if(mag_max < mag_ch[i]){
                        //if(mag_ch[i] > 32000){ //check for overflow
                        //    printf("x=%d, y=%d, mag_ch[%d] = %d \n", x, y, i, mag_ch[i]);
                        //}
                        mag_max = mag_ch[i]; //vectorized
                        mag_argmax = i;
                    }
                }
                #if 0 //real code
                int mag_max_sqrt = sqrt(mag_max);

                if(mag_max_sqrt > 256){ //2^17 = 362
                    printf("x=%d, y=%d, mag_max_sqrt = %d \n", x, y, mag_max_sqrt);
                }
                //vectorization falls down in the following lines:
                outMag[y*stride + x] = mag_max_sqrt;
                gradX = gradX_ch[mag_argmax];
                gradY = gradY_ch[mag_argmax];
                //outOri[y*stride + x] = ATAN2_TABLE[gradY + 255][gradX + 255]; //FIXME: this can be positive or negative
                outOri[y*stride + x] = abs(ATAN2_TABLE[gradY + 255][gradX + 255]) * 10; //for visual effect 
                #endif

                #if 1 //dummy code
                outOri[y*stride + x] = gradX_ch[0];
                //outOri[y*stride + x] = gradX_ch[0] + gradX_ch[1] + gradX_ch[2];
                //outOri[y*stride + x] = gradX_ch[mag_argmax];
                outMag[y*stride + x] = mag_max;
                #endif
            }
        }
    }
}

//TODO: make this much smaller than 512x512.
void init_lookup_table(){
    for (int dy = -255; dy <= 255; ++dy) { //pixels are 0 to 255, so gradient values are -255 to 255
        for (int dx = -255; dx <= 255; ++dx) {
            // Angle in the range [-pi, pi]
            double angle = atan2(static_cast<double>(dy), static_cast<double>(dx));

            // Convert it to the range [9.0, 27.0]
            angle = angle * (9.0 / M_PI) + 18.0;

            // Convert it to the range [0, 18)
            if (angle >= 18.0)
                angle -= 18.0;
            ATAN2_TABLE[dy + 255][dx + 255] = round( max(angle, 0.0) );
            //printf("ATAN2_TABLE[%d][%d] = %d \n", dx+255, dy+255, ATAN2_TABLE[dy + 255][dx + 255]);
        }
    }
}

int main (int argc, char **argv)
{
    init_lookup_table();
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
        gradient_sse(img.height, img.width, img.stride, img.n_channels, ori.n_channels, img.data, ori.data, mag.data); 
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


