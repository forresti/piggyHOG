#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cassert>

#include "SimpleImg.h"
#include "helpers.h"
using namespace std;
char ATAN2_TABLE[512][512]; //signed char (values are -18 to 18)

//no attempt at vectorization...just checking correctness.
void naive_fixedpt_gradient(int height, int width, int stride, int n_channels_input, int n_channels_output,
                            pixel_t *__restrict__ img, pixel_t *__restrict__ outOri, pixel_t *__restrict__ outMag){
    assert(n_channels_input == 3);
    assert(n_channels_output == 1);
    const int n_ch = 3; //hard-coded copy of n_channels_input

    //int mag_ch[3];
    int16_t mag_ch[3];
    int16_t gradX_ch[3];
    int16_t gradY_ch[3];
    int16_t ori_ch[3];

    int dataSize = 8; //long int
    

    for(int y=2; y<height-2; y++){
        //for(int x=2; x < width-2; x++){ //replaced with unrolling
        for(int x_tile=2; x_tile < (stride-dataSize); x_tile+=dataSize){
            //TODO: load a 8-byte long int for each grad_ch

            for(int x_inner=0; x_inner < dataSize; x_inner++){
                int x = x_tile + x_inner;
 
                gradX_ch[0] = (int16_t)img[y*stride + x +                 + 1] - (int16_t)img[y*stride + x                   - 1];
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
                #if 1 //real code
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

                #if 0 //dummy code
                //outOri[y*stride + x] = gradX_ch[0];
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

    //[mag, ori] = naive_fixedpt_gradient(img)
    SimpleImg mag(img.height, img.width, img.stride, 1); //out img has just 1 channel
    SimpleImg ori(img.height, img.width, img.stride, 1); //out img has just 1 channel
    double start_timer = read_timer();
    for(int i=0; i<n_iter; i++){
        naive_fixedpt_gradient(img.height, img.width, img.stride, img.n_channels, ori.n_channels, img.data, ori.data, mag.data); 
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


