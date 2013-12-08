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

#define eps 0.0001

//#define SCALE_ORI //if defined, scale up the orientation (1 to 18) to make it more visible in output images for debugging

//constructor
streamHog::streamHog(){
    init_atan2_constants(); //easier to vectorize alternative to lookup table. (similar to VOC5 hog)
    init_atan2_LUT(); //similar to FFLD hog
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
// ASSUMES that init_atan2_constants() has already been called.
void streamHog::init_atan2_LUT(){
    for (int dy = -255; dy <= 255; ++dy) { //pixels are 0 to 255, so gradient values are -255 to 255
        for (int dx = -255; dx <= 255; ++dx) {

            #if 0 //FFLD style, total of 19 bins [TODO: remove]
            // Angle in the range [-pi, pi]
            double angle = atan2(static_cast<double>(dy), static_cast<double>(dx));

            // Convert it to the range [9.0, 27.0]
            angle = angle * (9.0 / M_PI) + 18.0;

            // Convert it to the range [0, 18)
            if (angle >= 18.0)
                angle -= 18.0;
            ATAN2_TABLE[dy + 255][dx + 255] = round( max(angle, 0.0) );
            //printf("ATAN2_TABLE[%d][%d] = %d \n", dx+255, dy+255, ATAN2_TABLE[dy + 255][dx + 255]);
            #endif

            // snap to one of 18 orientations [VOC5 style]
            float best_dot = 0;
            int best_o = 0;
            for (int o = 0; o < 9; o++) {
                float dot = uu[o]*dx + vv[o]*dy;
                if (dot > best_dot) {
                    best_dot = dot;
                    best_o = o;
                }
                else if (-dot > best_dot) {
                    best_dot = -dot;
                    best_o = o+9;
                }
            }
            ATAN2_TABLE[dy + 255][dx + 255] = best_o;
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
//@return histogramBin( atan2(gradY, gradX) ) -- using approx atan2. 
//        input and output are packed int16_t
__m128i streamHog::approx_atan2_bin(__m128i gradX_max, __m128i gradY_max){

    __m128i isMax; //reset for each iteration
    __m128i best_dot = _mm_setzero_si128(); //max 
    __m128i best_ori = _mm_setzero_si128(); //argmax orientation

    __m128i negative_one_vec = _mm_set_epi16(-1, -1, -1, -1, -1, -1, -1, -1); //to calculate "-dot"
    __m128i nine_vec = _mm_set_epi16(9, 9, 9, 9, 9, 9, 9, 9);  //calculate "best_ori = ori+9;" when our best ori is negative

    // snap to one of 18 orientations
    for(int ori=0; ori<9; ori++){
        __m128i ori_vec = _mm_set_epi16(ori, ori, ori, ori, ori, ori, ori, ori); //copy of index for if/else in sse 
        __m128i dot = _mm_add_epi16( _mm_mullo_epi16(uu_fixedpt_epi16[ori], gradX_max),
                                     _mm_mullo_epi16(vv_fixedpt_epi16[ori], gradY_max) );

       
        //if(dot > best_dot){ best_dot = dot; best_ori = ori; } 
            isMax    = _mm_cmpgt_epi16(dot, best_dot); //comparing to OLD max from prev iteration
            best_dot = _mm_max_epi16(dot, best_dot);

            __m128i t0 = _mm_and_si128(ori_vec, isMax); //zero out nonmaxes in ori_vec
            __m128i t1 = _mm_andnot_si128(isMax, best_ori); //in ori argmaxes, zero out newly-beaten maxes 

            best_ori = _mm_or_si128(t0, t1);

        //if(-dot > best_dot){ best_dot = -dot; best_ori = ori+9; }
            __m128i minus_ori_vec = _mm_add_epi16(ori_vec, nine_vec); //ori+9
            __m128i minus_dot = _mm_mullo_epi16(dot, negative_one_vec); //-dot

            isMax = _mm_cmpgt_epi16(minus_dot, best_dot);  //if(-dot > best_dot)
            best_dot = _mm_max_epi16(minus_dot, best_dot); //    best_dot = -dot

            t0 = _mm_and_si128(minus_ori_vec, isMax); //zero out nonmaxes in ori_vec
            t1 = _mm_andnot_si128(isMax, best_ori); //in ori argmaxes, zero out newly-beaten maxes 

            best_ori = _mm_or_si128(t0, t1); 
    }

    #ifdef SCALE_ORI
    best_ori = _mm_mullo_epi16(best_ori, nine_vec); //TEST. scale up the ori vector...maybe it'll actually show up in pictures
    #endif 

    return best_ori;

}

//outOri_currPtr[0:15] = atan2(gradX_max[0:15], gradY_max[0:15])
// compute orientation from {gradX_max, gradY_max}. 
//  implemented as non-vectorized atan2 table lookup. 
// @param grad{X,Y}_max{0,1} = packed 16-bit gradients of max-mag channel
// @param outOri_currPtr = &outOri[y*stride + x]. 
void streamHog::ori_atan2_LUT(__m128i gradX_max_0, __m128i gradX_max_1, 
                              __m128i gradY_max_0, __m128i gradY_max_1, pixel_t* outOri_currPtr)
{
    //compute orientation from {gradX_max, gradY_max}. 
    //  implemented as non-vectorized atan2 table lookup. 
    int16_t gradX_max_unpacked[16]; //unpacked 8-bit numbers
    int16_t gradY_max_unpacked[16];

    _mm_store_si128( (__m128i*)(&gradX_max_unpacked[0]), gradX_max_0 ); //0:7
    _mm_store_si128( (__m128i*)(&gradX_max_unpacked[8]), gradX_max_1 ); //8:15
    _mm_store_si128( (__m128i*)(&gradY_max_unpacked[0]), gradY_max_0 ); //0:7
    _mm_store_si128( (__m128i*)(&gradY_max_unpacked[8]), gradY_max_1 ); //8:15

#if 1 //real code
    // non-vectorized atan2 table lookup.
    for(int i=0; i<16; i++){ 
        int16_t dx = gradX_max_unpacked[i];
        int16_t dy = gradY_max_unpacked[i];
        pixel_t ori = ATAN2_TABLE[dy+255][dx+255]; //ATAN2_TABLE is 0-18. (char)
        //pixel_t ori = ATAN2_TABLE[ (dy>>2) + 255 ][ (dx>>2) + 255 ]; //test LUT quantization -- looks fine.
        #ifdef SCALE_ORI
            ori = ori*9; //to be more visible in output images for debugging
        #endif
        outOri_currPtr[i] = ori; //outOri[y*stride + x + i] = ori;
    }
#endif 

#if 0 //very stripped down benchmark. (ignores the +/- overflow of 16-bit->8-bit for gradX,gradY)

    for(int i=0; i<16; i++){ 
        int16_t dx = gradX_max_unpacked[i];
        int16_t dy = gradY_max_unpacked[i];
        outOri_currPtr[i] = dy;
        //outOri_currPtr[i] = dx + dy;
    }
#endif

#if 0 //stripped down benchmark (just enough to force the compiler to compute gradX_max and gradY_max)

    __m128i gradX_max = _mm_packs_epi16(gradX_max_0, gradX_max_1); //16-bit -> 8-bit. (too low precision...just a test)
    __m128i gradY_max = _mm_packs_epi16(gradY_max_0, gradY_max_1);

    __m128i gradX_plus_gradY = _mm_add_epi8(gradX_max, gradY_max); //dummy work so that grad{X,Y}_max are computed
    _mm_store_si128( (__m128i*)(outOri_currPtr), gradX_plus_gradY );

    //_mm_store_si128( (__m128i*)(outOri_currPtr), gradX_max );
#endif

}

//TODO: replace outOri with outGradX_max and outGradY_max. (after calling gradient_stream, you do a lookup table)
//  or, just do the lookup in here...
void streamHog::gradient_stream(int height, int width, int stride, int n_channels_input, int n_channels_output,
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

    for(int y=0; y<height-2; y++){
        for(int x=0; x < stride-2; x+=loadSize){ //(stride-2) to avoid falling off the end when doing (location+2) to get xHi

            magMax = magMax_0 = magMax_1 = _mm_setzero_si128();

            for(int channel=0; channel<3; channel++){ //TODO: unroll channels
                //magIsArgmax_0_ch[channel] = magIsArgmax_1_ch[channel] = _mm_setzero_si128(); 

                xLo = _mm_loadu_si128( (__m128i*)(&img[(y+1)*stride + x + channel*height*stride    ]) ); //load sixteen 1-byte unsigned char pixels
                xHi = _mm_loadu_si128( (__m128i*)(&img[(y+1)*stride + x + channel*height*stride + 2]) ); //index as chars, THEN cast to __m128i*  

                //yLo = _mm_load_si128( (__m128i*)(&img[y*stride + x + channel*height*stride           ]) ); //y-dim is a long stride, easier to do aligned loads
                //yHi = _mm_load_si128( (__m128i*)(&img[y*stride + x + channel*height*stride + 2*stride]) );

                //TEST: fix off-by-one error. (need unaligned loads to fix it this way). slow?
                yLo = _mm_loadu_si128( (__m128i*)(&img[y*stride + (x+1) + channel*height*stride           ]) );
                yHi = _mm_loadu_si128( (__m128i*)(&img[y*stride + (x+1) + channel*height*stride + 2*stride]) );

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

                //gradX, gradY of the argmax(magnitude) channel
                select_epi16(mag_0_ch[channel], magMax_0, gradX_0_ch[channel], gradY_0_ch[channel], gradX_max_0, gradY_max_0); //output gradX_max_0, gradY_max_0
                select_epi16(mag_1_ch[channel], magMax_1, gradX_1_ch[channel], gradY_1_ch[channel], gradX_max_1, gradY_max_1); //output gradX_max_1, gradY_max_1 

                //magMax = max(mag_ch[0,1,2])
                magMax_0 = _mm_max_epi16(magMax_0, mag_0_ch[channel]);
                magMax_1 = _mm_max_epi16(magMax_1, mag_1_ch[channel]); 

                gradX_ch[channel] = _mm_packs_epi16(gradX_0_ch[channel], gradX_1_ch[channel]); //16-bit -> 8bit (temporary ... typically, we'd pack up the results later in the pipeline)
                gradY_ch[channel] = _mm_packs_epi16(gradY_0_ch[channel], gradY_1_ch[channel]); //temporary ... typically, we'd pack up the results later in the pipeline.

                //mag_ch[channel]   = _mm_packs_epi16(mag_0_ch[channel], mag_1_ch[channel]);
                //_mm_store_si128( (__m128i*)(&outOri[y*stride + x]), gradY_ch[channel] ); //outOri[y][x : x+15] = gradX_ch[channel] -- just a test, doesnt make much sense
                //_mm_store_si128( (__m128i*)(&outMag[y*stride + x]), mag_ch[channel] );
            }

            //TODO: shouldn't the output magnitudes be 16-bit? (to avoid overflow, e.g. (gradX=255 + gradY=255) > 255)
            //  or, rightshift (divide) the magnitude by 2...
            magMax = _mm_packs_epi16(magMax_0, magMax_1);
            _mm_store_si128( (__m128i*)(&outMag[y*stride + x]), magMax );
           
#if 1 //atan2 nonvectorized LUT. (not tested for correctness)
            //outOri[y*stride + x + 0:15] = atan2(gradX_max[0:15], gradY_max[0:15])
            ori_atan2_LUT(gradX_max_0, gradX_max_1, gradY_max_0, gradY_max_1, &outOri[y*stride + x]);
#endif
#if 0 //atan2 "snap-to ori" based on dot products, like VOC5. (not tested for correctness)
            __m128i oriMax_0 = approx_atan2_bin(gradX_max_0, gradY_max_0); //input and output packed int16_t
            __m128i oriMax_1 = approx_atan2_bin(gradX_max_1, gradY_max_1); 
            __m128i oriMax = _mm_packs_epi16(oriMax_0, oriMax_1); //16-bit -> 8-bit (values are 1 to 18)
            _mm_store_si128( (__m128i*)(&outOri[y*stride + x]), oriMax );
#endif
        }
    }
}

//gradient code from voc-release5 DPM. (reference impl)
void streamHog::gradient_voc5_reference(int height, int width, int stride, int n_channels_input, int n_channels_output,
                  pixel_t *__restrict__ img, pixel_t *__restrict__ outOri, pixel_t *__restrict__ outMag){

    for(int y=1; y<height-1; y++){
        for(int x=1; x<width-1; x++){

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
            channel=2;
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

            #ifdef SCALE_ORI
            best_o = best_o * 9;
            #endif
            //v = sqrt(v); //Forrest -- no longer need to sqrt the magnitude

            //outMag[y*stride + x] = v;
            //outOri[y*stride + x] = best_o;

            //to line up with forrest's 0-indexed version....
            outMag[(y-1)*stride + (x-1)] = v;
            outOri[(y-1)*stride + (x-1)] = best_o; 
        }
    }
}

//gradient code from voc-release5 DPM. (reference impl)
//  TODO: rearrange for the following out dims:
//assumed output dimensions: outHist[imgHeight/sbin][imgWidth/sbin][hogDepth=32]. row major (like piggyHOG).
//  output stride = output width. (because we already have 32-dimensional features as the inner dimension)
void streamHog::computeCells_voc5_reference(int imgHeight, int imgWidth, int imgStride, int sbin, 
                                            pixel_t *__restrict__ ori, pixel_t *__restrict__ mag,
                                            int outHistHeight, int outHistWidth,
                                            float *__restrict__ outHist){

    assert(outHistHeight == round(imgHeight/sbin));
    assert(outHistWidth == round(imgWidth/sbin));

    //TODO: have mag as an int16_t instead of a uchar. 

    const int hogDepth = 32;
    float sbin_inverse = 1.0f / (float)sbin;

#pragma omp parallel for
    for(int y=0; y<imgHeight; y++){
        for(int x=0; x<imgWidth; x++){
            int best_o = ori[y*imgStride + x]; //orientation bin -- upcast to int
            int v = mag[y*imgStride + x]; //upcast to int

            // add to 4 histograms around pixel using linear interpolation
            //float xp = ((float)x+0.5)/(float)sbin - 0.5; //this is expensive (replacing it with 'x/4' gives a 1.5x speedup in hogCell)
            //loat yp = ((float)y+0.5)/(float)sbin - 0.5;
            //float xp = x/4; //TEST
            //float yp = y/4;
            float xp = ((float)x+0.5)*sbin_inverse - 0.5;
            float yp = ((float)y+0.5)*sbin_inverse - 0.5;
            int ixp = (int)floor(xp);
            int iyp = (int)floor(yp);
            float vx0 = xp-ixp;
            float vy0 = yp-iyp;
            float vx1 = 1.0-vx0;
            float vy1 = 1.0-vy0;

            //if(ixp < 0 || iyp < 0){
            //    printf("x=%d, y=%d, ixp=%d, iyp=%d \n", x, y, ixp, iyp);
            //}

            if (ixp >= 0 && iyp >= 0) 
            { 
                //*(hist + ixp*imgHeight + iyp + best_o*imgHeight*imgWidth) += vx1*vy1*v; //from VOC5

                outHist[ixp*hogDepth + iyp*outHistWidth*hogDepth + best_o] += vx1*vy1*v; //[5.6ms on laptop, 640x480, 1 level, sbin=4, only accumulate to 1 cell]
                //outHist[ixp*hogDepth + iyp*outHistWidth*hogDepth + best_o] += v; //test -- avoid multiplying by weights [5.5ms on laptop]
                //outHist[ixp*hogDepth + iyp*outHistWidth*hogDepth] += v; //test -- always use bin 0. [3.2ms on laptop]
            } 

#if 1
            if (ixp+1 < outHistWidth && iyp >= 0) { 
                //*(hist + (ixp+1)*imgHeight + iyp + best_o*imgHeight*imgWidth) += vx0*vy1*v;

                outHist[(ixp+1)*hogDepth + iyp*outHistWidth*hogDepth + best_o] += vx0*vy1*v;
            } 

            if (ixp >= 0 && iyp+1 < outHistHeight) { 
                //*(hist + ixp*imgHeight + (iyp+1) + best_o*imgHeight*imgWidth) += vx1*vy0*v;

                outHist[ixp*hogDepth + (iyp+1)*outHistWidth*hogDepth + best_o] += vx1*vy0*v;
            } 

            if (ixp+1 < outHistWidth && iyp+1 < outHistHeight) { 
                //*(hist + (ixp+1)*imgHeight + (iyp+1) + best_o*imgHeight*imgWidth) += vx0*vy0*v;

                outHist[(ixp+1)*hogDepth + (iyp+1)*outHistWidth*hogDepth + best_o] += vx0*vy0*v;
            } 
#endif
        }
    }
}

//start from stream benchmark, gradually build up the histogram code.
void streamHog::computeCells_stream(int imgHeight, int imgWidth, int imgStride, int sbin,
                                    pixel_t *__restrict__ ori, pixel_t *__restrict__ mag,
                                    int outHistHeight, int outHistWidth,
                                    float *__restrict__ outHist){

    const int hogDepth = 32;
    float sbin_inverse = 1.0f / (float)sbin;

    float pos_LUT[sbin]; //xp,yp ... access as pos_LUT[x%sbin]
    int ipos_LUT[sbin]; //ixp, iyp
    float v0_LUT[sbin]; //vx0, vy0
    float v1_LUT[sbin]; //vx1, vy1

    //main idea: xp, yp are cyclic. so, just cache the cycle, and later on add the x,y offset.
    for(int i=0; i<sbin; i++){
        pos_LUT[i] = ((float)(i%sbin)+0.5)/(float)sbin - 0.5;
        ipos_LUT[i] = floor(pos_LUT[i]);
        v0_LUT[i] = pos_LUT[i] - ipos_LUT[i]; //xp-ixp
        v1_LUT[i] = 1.0f - v0_LUT[i]; //1.0-vx0
    }

    //TODO: x=2:imgHeight-2. no checking for 'if ixp>=0,...' 
    //      then, have a final 'cleanup' loop for the x=0:2, x=imgHeight-2:imgHeight, etc.
    //      same deal for y. 

#pragma omp parallel for
    for(int y=0; y<imgHeight; y++){
        for(int x=0; x < imgWidth; x++){
            int curr_ori = ori[y*imgStride + x]; //orientation bin -- upcast to int
            int curr_mag = mag[y*imgStride + x]; //upcast to int

#if 0
            float xp = ((float)x+0.5)*sbin_inverse - 0.5;
            float yp = ((float)y+0.5)*sbin_inverse - 0.5;
            int ixp = (int)floor(xp);
            int iyp = (int)floor(yp);
            float vx0 = xp-ixp;
            float vy0 = yp-iyp;
            float vx1 = 1.0-vx0;
            float vy1 = 1.0-vy0;
#endif
#if 1
            //tested: my LUT matches VOC5 
            int ixp = ipos_LUT[x%sbin] + floor(x*sbin_inverse);
            int iyp = ipos_LUT[y%sbin] + floor(y*sbin_inverse);
            float vx0 = v0_LUT[x%sbin];
            float vy0 = v0_LUT[y%sbin];
            float vx1 = v1_LUT[x%sbin];
            float vy1 = v1_LUT[y%sbin];
#endif
            //temporary: (will do something more clever later) -- this is SLOW and not quite right at edges
            //ixp = max(0, ixp);
            //iyp = max(0, iyp);
            //ixp = min(ixp, outHistWidth-2);
            //iyp = min(iyp, outHistHeight-2);

            //TODO: instead of doing 'if (ixp >= 0 && iyp >= 0)' on every iteration,
            //      have a separate set of loops for border cases

//TODO: atomic add if race condition shows up
          //the actual computation:
            if (ixp >= 0 && iyp >= 0) //this is expensive. 
            {
                //outHist[x_hist*hogDepth + y_hist*outHistWidth*hogDepth + 0] = curr_mag; //simple benchmark [2.6 GB/s = .91ms on laptop]
                //outHist[x_hist*hogDepth + y_hist*outHistWidth*hogDepth + curr_ori] = curr_mag; //[2.1 GB/s = 1.15ms on laptop]
                //outHist[ixp*hogDepth + iyp*outHistWidth*hogDepth + 0] = curr_mag*vx1*vy1;
                //outHist[ixp*hogDepth + iyp*outHistWidth*hogDepth + curr_ori] += curr_mag;
                outHist[ixp*hogDepth + iyp*outHistWidth*hogDepth + curr_ori] += vx1*vy1*curr_mag;
            }
            if (ixp+1 < outHistWidth && iyp >= 0) {
                outHist[(ixp+1)*hogDepth + iyp*outHistWidth*hogDepth + curr_ori] += vx0*vy1*curr_mag;
            }
            if (ixp >= 0 && iyp+1 < outHistHeight) {
                outHist[ixp*hogDepth + (iyp+1)*outHistWidth*hogDepth + curr_ori] += vx1*vy0*curr_mag;
            }
            if (ixp+1 < outHistWidth && iyp+1 < outHistHeight) {
                outHist[(ixp+1)*hogDepth + (iyp+1)*outHistWidth*hogDepth + curr_ori] += vx0*vy0*curr_mag;
            }
            //DEBUG printfs.
            //float my_outHist_element = outHist[ixp*hogDepth + iyp*outHistWidth*hogDepth + curr_ori];
            //if(my_outHist_element > 512)
            {
//                printf("outHist[ixp=%d][iyp=%d][curr_ori=%d] = %f \n", ixp, iyp, curr_ori, my_outHist_element);
            }
        }
    } 
}

//TODO: handle hog padding. (for now, just use padded HOG dims as input here)
//@in-out normImg = 1-channel img of size (histWidth x histHeight). we populate this with hist sums.
void streamHog::hogCell_gradientEnergy(float *__restrict__ hogHist, int histHeight, int histWidth, 
                                       float *__restrict__ normImg)
{
    const int hogDepth = 32;

    //sum up the (0 to 360 degree) hog cells
    for(int y=0; y < histHeight; y++){
        for(int x=0; x < histWidth; x++){

            int hogIdx = x*hogDepth + y*histWidth*hogDepth;
            float norm = 0.0f;
            for(int ori=0; ori<18; ori++){ //0-360 degree ori bins
                norm += hogHist[hogIdx+ori] * hogHist[hogIdx+ori]; //squared -- will do sqrt in hogBlock_normalize()

                //TODO: can we do abs() instead of squaring this?
            }
            normImg[y*histHeight + x] = norm;
        }
    }
}

//TODO: compute 9 contrast-insensitive features

//hog cells -> hog blocks
//@param in_hogHist = hog cells
//@param in_normImg = result of hogCell_gradientEnergy()
//@param out_hogBlocks = normalized hog blocks
void streamHog::normalizeCells_voc5(float *__restrict__ in_hogHist, float *__restrict__ in_normImg, 
                                    float *__restrict__ out_hogBlocks,
                                    int histHeight, int histWidth)
{
    const int hogDepth = 32;
    //TODO: test my ptr indexing vs. voc5 ptr indexing.

#if 1
    // compute features
#pragma omp parallel for
    for(int y=1; y < histHeight-1; y++){
        for(int x=1; x < histWidth-1; x++){

            #if 0 //VOC5
            float *dst = feat + x*out[0] + y;
            float *src, *p, n1, n2, n3, n4;
            p = norm + (x+1)*blocks[0] + y+1;
            n1 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[0]) + *(p+blocks[0]+1) + eps);
            p = norm + (x+1)*blocks[0] + y;
            n2 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[0]) + *(p+blocks[0]+1) + eps);
            p = norm + x*blocks[0] + y+1;
            n3 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[0]) + *(p+blocks[0]+1) + eps);
            p = norm + x*blocks[0] + y;
            n4 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[0]) + *(p+blocks[0]+1) + eps);
            #endif

            #if 1 //Forrest
            float n1 = 1.0 / sqrt(in_normImg[(y-1)*histWidth + (x-1)] + //top-left
                                  in_normImg[(y-1)*histWidth + (x)]   +
                                  in_normImg[(y)*histWidth   + (x-1)] +
                                  in_normImg[(y)*histWidth   + (x)]   + eps);

            float n2 = 1.0 / sqrt(in_normImg[(y-1)*histWidth + (x)]   + //top-right
                                  in_normImg[(y-1)*histWidth + (x+1)] +
                                  in_normImg[(y)*histWidth   + (x)]   +
                                  in_normImg[(y)*histWidth   + (x+1)] + eps);

            float n3 = 1.0 / sqrt(in_normImg[(y)*histWidth   + (x-1)] + //bottom-left
                                  in_normImg[(y)*histWidth   + (x)]   +
                                  in_normImg[(y+1)*histWidth + (x-1)] +
                                  in_normImg[(y+1)*histWidth + (x)]   + eps);

            float n4 = 1.0 / sqrt(in_normImg[(y)*histWidth   + (x)]   + //bottom-right
                                  in_normImg[(y)*histWidth   + (x+1)] +
                                  in_normImg[(y+1)*histWidth + (x)]   +
                                  in_normImg[(y+1)*histWidth + (x+1)] + eps);
            #endif

            float t1 = 0;
            float t2 = 0;
            float t3 = 0;
            float t4 = 0;

            #if 0 //VOC5
            // contrast-sensitive features
            //src = hist + (x+1)*blocks[0] + (y+1);
            src = hist + (x+1) + (y+1)*histWidth;
            for (int o = 0; o < 18; o++) {
                float h1 = min(*src * n1, 0.2);
                float h2 = min(*src * n2, 0.2);
                float h3 = min(*src * n3, 0.2);
                float h4 = min(*src * n4, 0.2);
                *dst = 0.5 * (h1 + h2 + h3 + h4);
                t1 += h1;
                t2 += h2;
                t3 += h3;
                t4 += h4;
                dst += out[0]*out[1];
                src += blocks[0]*blocks[1];
            }
            #endif

            #if 1 //Forrest
            // contrast-sensitive features
            //src = hist + (x+1)*blocks[0] + (y+1);
            int hogIdx = (x+1)*hogDepth + (y+1)*histWidth*hogDepth; 
            for (int o = 0; o < 18; o++) {
                float in_bin = in_hogHist[hogIdx + o]; 

                float h1 = min(in_bin * n1, 0.2f);
                float h2 = min(in_bin * n2, 0.2f);
                float h3 = min(in_bin * n3, 0.2f);
                float h4 = min(in_bin * n4, 0.2f);
               
                //*dst = 0.5 * (h1 + h2 + h3 + h4); 
                out_hogBlocks[hogIdx + o] = 0.5f * (h1 + h2 + h3 + h4);

                t1 += h1;
                t2 += h2;
                t3 += h3;
                t4 += h4;
                //dst += out[0]*out[1];
                //src += blocks[0]*blocks[1];
            }

            // contrast-insensitive features
            //src = hist + (x+1)*blocks[0] + (y+1);
            for (int o = 0; o < 9; o++) {
                //float sum = *src + *(src + 9*blocks[0]*blocks[1]);
                float sum = in_hogHist[hogIdx + o] + in_hogHist[hogIdx+ o+9]; 
                float h1 = min(sum * n1, 0.2f);
                float h2 = min(sum * n2, 0.2f);
                float h3 = min(sum * n3, 0.2f);
                float h4 = min(sum * n4, 0.2f);
                //*dst = 0.5 * (h1 + h2 + h3 + h4);
                //dst += out[0]*out[1];
                //src += blocks[0]*blocks[1];
                out_hogBlocks[hogIdx + o + 18] = sum;
            }
            out_hogBlocks[hogIdx + 27] = 0.2357f * t1;
            out_hogBlocks[hogIdx + 28] = 0.2357f * t2;
            out_hogBlocks[hogIdx + 29] = 0.2357f * t3;
            out_hogBlocks[hogIdx + 30] = 0.2357f * t4;
            #endif
        }
    }
#endif
}


