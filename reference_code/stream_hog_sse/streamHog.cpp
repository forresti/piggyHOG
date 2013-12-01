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

//constructor
streamHog::streamHog(){
    init_lookup_table(); //similar to FFLD hog
    init_atan2_constants(); //easier to vectorize alternative to lookup table. (similar to VOC5 hog)
}

//destructor
streamHog::~streamHog(){ }

//stuff for approximate vectorized atan2
void streamHog::init_atan2_constants(){
    double  uu_local[9] = {1.0000, 0.9397, 0.7660, 0.500, 0.1736, -0.1736, -0.5000, -0.7660, -0.9397}; //from voc-release5 features.cc
    double  vv_local[9] = {0.0000, 0.3420, 0.6428, 0.8660, 0.9848, 0.9848, 0.8660, 0.6428, 0.3420}; 

    for(int i=0; i<9; i++){
        uu[i] = uu_local[i]; //can't do array[9]={data, data, ...} initialization in a class. 
        vv[i] = vv_local[i];

        uu_fixedpt[i] = round(uu[i] * 100);
        vv_fixedpt[i] = round(vv[i] * 100);

        //vector of copies of uu and vv for SSE vectorization
        uu_fixedpt_epi16[i] = _mm_set_epi16(uu_fixedpt[i],uu_fixedpt[i],uu_fixedpt[i],uu_fixedpt[i],uu_fixedpt[i],uu_fixedpt[i],uu_fixedpt[i],uu_fixedpt[i]); 
        vv_fixedpt_epi16[i] = _mm_set_epi16(vv_fixedpt[i],vv_fixedpt[i],vv_fixedpt[i],vv_fixedpt[i],vv_fixedpt[i],vv_fixedpt[i],vv_fixedpt[i],vv_fixedpt[i]); 
    }
}

//TODO: make this much smaller than 512x512.
void streamHog::init_lookup_table(){
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

//enables us to load 8-bit values, but work in 16-bit. 
void streamHog::upcast_8bit_to_16bit(__m128i in_xLo,  __m128i in_xHi, __m128i in_yLo,     __m128i in_yHi,
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
}

//@param magChannel = current channel's magnitude
//@param old_magMax = maximum gradient *seen by previous iteration* (note: this code *doesn't* update old_magMax)
//@in-out gradX_max, gradY_max = output gradient of max channel (of the channels checked so far)
void streamHog::select_epi16(__m128i magChannel, __m128i old_magMax, 
                             __m128i gradX_channel, __m128i gradY_channel,
                             __m128i &gradX_max, __m128i &gradY_max){

    //print_epi16(gradX_max, "gradX_max prev iteration");

    __m128i isMax = _mm_cmpgt_epi16(magChannel, old_magMax); // = 1 when magChannel is max that we have seen so far

    //if magChannel is max, gradX_channel_tmp=gradX_channel; 
    //else                  gradX_channel_tmp = 0
    __m128i gradX_channel_tmp = _mm_and_si128(gradX_channel, isMax); //zero out non-maxes from this channel
    __m128i gradY_channel_tmp = _mm_and_si128(gradY_channel, isMax); 

    //if magChannel is NOT max, gradX_max_tmp = gradX_max
    //else                      gradX_max_tmp = 0
    __m128i gradX_max_tmp = _mm_andnot_si128(isMax, gradX_max); //zero out non-maxes from previous channels
    __m128i gradY_max_tmp = _mm_andnot_si128(isMax, gradY_max);
   
    gradX_max = _mm_or_si128(gradX_channel_tmp, gradX_max_tmp); //for each element, ONE of these 2 args is nonzero
    gradY_max = _mm_or_si128(gradY_channel_tmp, gradY_max_tmp); 

    //print_epi16(isMax, "isMax");
    //print_epi16(gradX_max_tmp, "gradX_max_tmp");
    //print_epi16(gradX_max, "gradX_max");
}

//@param  gradX_max, gradY_max = output gradient of max channel (of the channels checked so far)
//@return histogramBin( atan2(gradY, gradX) ) -- using approx atan2
__m128i streamHog::approx_atan2_bin(__m128i gradX_max, __m128i gradY_max){

    __m128i isMax; //reset for each iteration
    __m128i best_dot = _mm_setzero_si128(); //max 
    __m128i best_ori = _mm_setzero_si128(); //argmax orientation

    __m128i negative_one_vec = _mm_set_epi16(-1, -1, -1, -1, -1, -1, -1, -1); //to calculate "-dot"
    __m128i nine_vec = _mm_set_epi16(9, 9, 9, 9, 9, 9, 9, 9);  //calculate "best_ori = ori+9;" when our best ori is negative

    // snap to one of 18 orientations
    for(int ori=0; ori<9; ori++){
#if 0
        __m128i ori_vec = _mm_set_epi16(ori, ori, ori, ori, ori, ori, ori, ori); //copy of index for if/else in sse 
        __m128i dot = _mm_add_epi16( _mm_mullo_epi16(uu_fixedpt_epi16[ori], gradX_max),
                                     _mm_mullo_epi16(vv_fixedpt_epi16[ori], gradY_max) );

       
        //if(dot > best_dot){ best_dot = dot; best_ori = ori;} 
            isMax    = _mm_cmpgt_epi16(dot, best_dot); //comparing to OLD max from prev iteration
            best_dot = _mm_max_epi16(dot, best_dot);

            __m128i t0 = _mm_and_si128(ori_vec, isMax); //zero out nonmaxes in ori_vec
            __m128i t1 = _mm_andnot_si128(isMax, best_ori); //in ori argmaxes, zero out newly-beaten maxes 

            best_dot = _mm_or_si128(t0, t1);

        //if(-dot > best_dot){ best_dot = -dot; best_ori = ori+9; }
        //TODO: implement -best_dot
            __m128i minus_ori_vec = _mm_add_epi16(ori_vec, nine_vec); //ori+9
            __m128i minus_dot = _mm_mul_epi16(dot, negative_one_vec); //-dot

            isMax = _mm_cmpgt_epi16(minus_dot, best_dot);  //if(-dot > best_dot)
            best_dot = _mm_max_epi16(minus_dot, best_dot); //    best_dot = -dot

            __m128i t0 = _mm_and_si128(minus_ori_vec, isMax); //zero out nonmaxes in ori_vec
            __m128i t1 = _mm_andnot_si128(isMax, best_ori); //in ori argmaxes, zero out newly-beaten maxes 

            best_dot = _mm_or_si128(t0, t1); 

        //TODO: in voc5 code, check whether the 'negative ori' case NEEDS to be an "else if." (try running with just "if.") 
#endif
    }

    return best_ori;

}

void streamHog::gradient_sse(int height, int width, int stride, int n_channels_input, int n_channels_output,
                  pixel_t *__restrict__ img, pixel_t *__restrict__ outOri, pixel_t *__restrict__ outMag){
    assert(n_channels_input == 3);
    assert(n_channels_output == 1);
    assert(sizeof(__m128i) == 16);
    int loadSize = sizeof(__m128i); // 16 bytes = 128 bits

    //input pixels
    __m128i xLo, xHi, yLo, yHi; //packed 8-bit
    __m128i xLo_0, xHi_0, yLo_0, yHi_0; //bottom bits: upcast from 8-bit to 16-bit
    __m128i xLo_1, xHi_1, yLo_1, yHi_1; //top bits: upcast from 8-bit to 16-bit

    //gradients
    __m128i gradX_ch[3],   gradY_ch[3];   //packed 8-bit
    __m128i gradX_0_ch[3], gradY_0_ch[3]; //bottom bits: upcast from 8-bit to 16-bit
    __m128i gradX_1_ch[3], gradY_1_ch[3]; //top bits: upcast from 8-bit to 16-bit
    __m128i gradX_max_0, gradX_max_1; //gradX of the max-mag channel
    __m128i gradY_max_0, gradY_max_1; //gradY of the max-mag channel
    __m128i gradMax_0, gradMax_1; //bottom bits, top bits (after arctan)

    //magnitudes
    __m128i mag_ch[3]; //packed 8-bit
    __m128i mag_0_ch[3]; //bottom bits
    __m128i mag_1_ch[3]; //top bits
    __m128i magMax, magMax_0, magMax_1; //packed 8-bit, bottom bits, top bits
    //__m128i magIsArgmax_0_ch[3], magIsArgmax_1_ch[3]; //bottom bits, top bits. boolean bitmask for "is this channel the mag argmax?"

    for(int y=2; y<height-2; y++){
        for(int x=0; x < stride-2; x+=loadSize){ //(stride-2) to avoid falling off the end when doing (location+2) to get xHi

            magMax = magMax_0 = magMax_1 = _mm_setzero_si128();

            for(int channel=0; channel<3; channel++){ //TODO: unroll channels
                //magIsArgmax_0_ch[channel] = magIsArgmax_1_ch[channel] = _mm_setzero_si128(); 

                xLo = _mm_loadu_si128( (__m128i*)(&img[y*stride + x + channel*height*stride    ]) ); //load sixteen 1-byte unsigned char pixels
                xHi = _mm_loadu_si128( (__m128i*)(&img[y*stride + x + channel*height*stride + 2]) ); //index as chars, THEN cast to __m128i*  

                yLo = _mm_load_si128( (__m128i*)(&img[y*stride + x + channel*height*stride           ]) ); //y-dim is a long stride, easier to do aligned loads
                yHi = _mm_load_si128( (__m128i*)(&img[y*stride + x + channel*height*stride + 2*stride]) );
                upcast_8bit_to_16bit(xLo, xHi, yLo, yHi,
                                     xLo_0, xHi_0, yLo_0, yHi_0,
                                     xLo_1, xHi_1, yLo_1, yHi_1);

                gradX_0_ch[channel] =  _mm_sub_epi16(xHi_0, xLo_0); // xHi[0:7] - xLo[0:7]
                gradX_1_ch[channel] =  _mm_sub_epi16(xHi_1, xLo_1); // xHi[8:15] - xLo[8:15]

                gradY_0_ch[channel] =  _mm_sub_epi16(yHi_0, yLo_0);
                gradY_1_ch[channel] =  _mm_sub_epi16(yHi_1, yLo_1); 

                //mag = abs(gradX) + abs(gradY)
                // this is using the non-sqrt approach that has proved equally accurate to mag=sqrt(gradX^2 + gradY^2)
                mag_0_ch[channel] = _mm_add_epi16( _mm_abs_epi16(gradX_0_ch[channel]), _mm_abs_epi16(gradY_0_ch[channel]) ); // abs(gradX[0:7]) + abs(gradY[0:7])
                mag_1_ch[channel] = _mm_add_epi16( _mm_abs_epi16(gradX_1_ch[channel]), _mm_abs_epi16(gradY_1_ch[channel]) ); // abs(gradX[8:15]) + abs(gradY[8:15])
                mag_ch[channel]   = _mm_packs_epi16(mag_0_ch[channel], mag_1_ch[channel]);

                //gradX, gradY of the argmax(magnitude) channel
                    //TODO: test this
                select_epi16(mag_0_ch[channel], magMax_0, gradX_0_ch[channel], gradY_0_ch[channel], gradX_max_0, gradY_max_0); //output gradX_max_0, gradY_max_0
                select_epi16(mag_1_ch[channel], magMax_1, gradX_1_ch[channel], gradY_1_ch[channel], gradX_max_1, gradY_max_1); //output gradX_max_1, gradY_max_1 

                //TODO: compute approx atan2 of (gradY_max_0, gradX_max_0) in fixed pt
                //      use _mm_mullo_epi16() to calculate tangents

                //magMax = max(mag_ch[0,1,2])
                magMax_0 = _mm_max_epi16(magMax_0, mag_0_ch[channel]);
                magMax_1 = _mm_max_epi16(magMax_1, mag_1_ch[channel]); 

                gradX_ch[channel] = _mm_packs_epi16(gradX_0_ch[channel], gradX_1_ch[channel]); //16-bit -> 8bit (temporary ... typically, we'd pack up the results later in the pipeline)
                gradY_ch[channel] = _mm_packs_epi16(gradY_0_ch[channel], gradY_1_ch[channel]); //temporary ... typically, we'd pack up the results later in the pipeline.


                _mm_store_si128( (__m128i*)(&outOri[y*stride + x]), gradX_ch[channel] ); //outOri[y][x : x+15] = gradX_ch[channel] -- just a test, doesnt make much sense
                //_mm_store_si128( (__m128i*)(&outMag[y*stride + x]), gradY_ch[channel] ); //aligned stores are easy here...it's all divisible by loadSize.
                //_mm_store_si128( (__m128i*)(&outMag[y*stride + x]), mag_ch[channel] );
            }

            magMax = _mm_packs_epi16(magMax_0, magMax_1);
            _mm_store_si128( (__m128i*)(&outMag[y*stride + x]), magMax );
        }
    }
}

// gives not-quite-right results:
//no attempt at vectorization...just checking correctness.
void streamHog::gradient_wideload_unvectorized(int height, int width, int stride, int n_channels_input, int n_channels_output,
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

//gradient code from voc-release5 DPM. (reference impl)
void streamHog::gradient_voc5_reference(int height, int width, int stride, int n_channels_input, int n_channels_output,
                  pixel_t *__restrict__ img, pixel_t *__restrict__ outOri, pixel_t *__restrict__ outMag){

    for(int y=2; y<height-2; y++){
        for(int x=2; x<width-2; x++){

            int channel = 0;
            double dx = (double)img[y*stride + (x+1) + channel*height*stride] - 
                            (double)img[y*stride + (x-1) + channel*height*stride];
            double dy = (double)img[(y+1)*stride + x + channel*height*stride] -
                            (double)img[(y-1)*stride + x + channel*height*stride];
            //double v = dx*dx + dy*dy; //max magnitude (gets updated later)
            double v = fabs(dx) + fabs(dy);

            // second color channel
            channel=1;
            double dx2 = (double)img[y*stride + (x+1) + channel*height*stride] -
                            (double)img[y*stride + (x-1) + channel*height*stride];
            double dy2 = (double)img[(y+1)*stride + x + channel*height*stride] -
                            (double)img[(y-1)*stride + x + channel*height*stride];
            //double v2 = dx2*dx2 + dy2*dy2;
            double v2 = fabs(dx2) + fabs(dy2);

            // third color channel
            double dx3 = (double)img[y*stride + (x+1) + channel*height*stride] -
                            (double)img[y*stride + (x-1) + channel*height*stride];
            double dy3 = (double)img[(y+1)*stride + x + channel*height*stride] -
                            (double)img[(y-1)*stride + x + channel*height*stride];
            //double v3 = dx3*dx3 + dy3*dy3;
            double v3 = fabs(dx3) + fabs(dy3); //Forrest's version

            // pick channel with strongest gradient
            if (v2 > v) {
                v = v2; 
                dx = dx2;
                dy = dy2;
            }     
            if (v3 > v) {
                v = v3; 
                dx = dx3;
                dy = dy3;
            }     

            // snap to one of 18 orientations
            double best_dot = 0;
            int best_o = 0;
            for (int o = 0; o < 9; o++) {
                double dot = uu[o]*dx + vv[o]*dy;
                if (dot > best_dot) {
                    best_dot = dot;
                    best_o = o;
                } else if (-dot > best_dot) {
                    best_dot = -dot;
                    best_o = o+9;
                }   
            }

            outMag[y*stride + x] = v;
            outOri[y*stride + x] = best_o; 
        }
    }
}

