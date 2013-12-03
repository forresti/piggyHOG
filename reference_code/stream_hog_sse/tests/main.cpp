#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "helpers.h"
using namespace std;


// unit vectors used to compute gradient orientation
double  uu[9] = {1.0000, 0.9397, 0.7660, 0.500, 0.1736, -0.1736, -0.5000, -0.7660, -0.9397};
double  vv[9] = {0.0000, 0.3420, 0.6428, 0.8660, 0.9848, 0.9848, 0.8660, 0.6428, 0.3420};
int16_t uu_fixedpt[9]; //scalar fixed-pt (scaled up by 100)
int16_t vv_fixedpt[9];
__m128i uu_fixedpt_epi16[9]; //each of these vectors is bunch of copies of uu_fixedpt[i]
__m128i vv_fixedpt_epi16[9];

//alternatuve to unit vectors -- FFLD-style lookup table
char ATAN2_TABLE[512][512]; //signed char (values are -18 to 18)

//stuff for approximate vectorized atan2
void init_atan2_constants(){
    for(int i=0; i<9; i++){
        uu_fixedpt[i] = round(uu[i] * 100.0f);
        vv_fixedpt[i] = round(vv[i] * 100.0f);

//        printf("uu[%d]=%f, uu_fixedpt[%d]=%d \n", i, uu[i], i, uu_fixedpt[i]);
//        printf("vv[%d]=%f, vv_fixedpt[%d]=%d \n", i, vv[i], i, vv_fixedpt[i]);

        //vector of copies of uu and vv for SSE vectorization
        uu_fixedpt_epi16[i] = _mm_set_epi16(uu_fixedpt[i],uu_fixedpt[i],uu_fixedpt[i],uu_fixedpt[i],uu_fixedpt[i],uu_fixedpt[i],uu_fixedpt[i],uu_fixedpt[i]);
        vv_fixedpt_epi16[i] = _mm_set_epi16(vv_fixedpt[i],vv_fixedpt[i],vv_fixedpt[i],vv_fixedpt[i],vv_fixedpt[i],vv_fixedpt[i],vv_fixedpt[i],vv_fixedpt[i]);
    }
}

//FFLD-style lookup table
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

//@in-out ori_best_bin[dx][dy] = atan2 ori bin that best matches the gradient (dx,dy) 
//@in-out ori_dot[dx][dy][bin] = how well we match with each orientation bin
//uses FIXED-point calculations. (this is the reference implementation from voc-release5)
//void atan2_snap_to_fixedpt(int* ori_best_bin, int16_t* ori_dot)
void atan2_snap_to_fixedpt(int* ori_best_bin, int* ori_dot)
{

    for(int16_t dx=-255; dx <= 255; dx++){
        for(int16_t dy=-255; dy <= 255; dy++){

            int16_t best_dot = 0;
            //int best_dot = 0; //TEST -- use int32 instead of int16
            int best_o = 0;

            for (int o = 0; o < 9; o++) {
                //double dot = uu[o]*dx + vv[o]*dy; //float

                int16_t dot = uu_fixedpt[o]*dx + vv_fixedpt[o]*dy;
                //int16_t dot = uu_fixedpt[o]*dx; //simplify (TODO: remove)
                //int dot = (int)uu_fixedpt[o]*(int)dx + (int)vv_fixedpt[o]*(int)dy;  //TEST -- use int32 instead of int16

                ori_dot[ (dx+255)*512*18 + (dy+255)*18 + o] = dot;
                ori_dot[ (dx+255)*512*18 + (dy+255)*18 + (o+9)] = -dot;

                if (dot > best_dot) {
                    best_dot = dot;
                    best_o = o;
                } else if (-dot > best_dot) {
                    best_dot = -dot;
                    best_o = o+9;
                }

                ori_best_bin[ (dx+255)*512 + (dy*255) ] = best_o;
            }
        }
    }
}

//@in-out ori_best_bin[dx][dy] = atan2 ori bin that best matches the gradient (dx,dy) 
//@in-out ori_dot[dx][dy][bin] = how well we match with each orientation bin
//uses FLOATING-point calculations. (this is the reference implementation from voc-release5)
void atan2_snap_to_floatpt(int* ori_best_bin, float* ori_dot){

    for(int16_t dx=-255; dx <= 255; dx++){
        for(int16_t dy=-255; dy <= 255; dy++){

            float best_dot = 0;
            int best_o = 0;

            for (int o = 0; o < 9; o++) {
                float dot = uu[o]*(float)dx + vv[o]*(float)dy; //float
                //float dot = uu[o] * (double)dx; //simplify (TODO: remove)

                ori_dot[ (dx+255)*512*18 + (dy+255)*18 + o] = dot;
                ori_dot[ (dx+255)*512*18 + (dy+255)*18 + (o+9)] = -dot;

                if (dot > best_dot) {
                    best_dot = dot;
                    best_o = o;
                } else if (-dot > best_dot) {
                    best_dot = -dot;
                    best_o = o+9;
                }

                ori_best_bin[ (dx+255)*512 + (dy*255) ] = best_o;
            }
        }
    }
}

void test_int16_range(){

    for(int ex=2; ex<20; ex++){
        int tmp_32 = 2 << ex;
        int16_t tmp_16 = 2 << ex;

        printf("2<<%d. int16: %d, int32: %d \n", ex, tmp_16, tmp_32);
    }

    int tmp_32 = -32767;
    int16_t tmp_16 = tmp_32;
    printf("%d. int16: %d, int32: %d \n", tmp_32, tmp_16, tmp_32);
}

void fixedpt_vs_floatpt(){

    //test_int16_range();
    init_atan2_constants(); //stuff for fixedpt
    init_lookup_table();    

  //floating-pt "best ori bin" experiment
    int* ori_best_bin_floatpt = (int*)malloc(512*512 * sizeof(int));
    float* ori_dot_floatpt = (float*)malloc(512*512*18 * sizeof(float));
    atan2_snap_to_floatpt(ori_best_bin_floatpt, ori_dot_floatpt);

  //fixed-pt "best ori bin" experiment
    int* ori_best_bin_fixedpt = (int*)malloc(512*512 * sizeof(int));
    int* ori_dot_fixedpt = (int*)malloc(512*512*18 * sizeof(int));
    //int16_t* ori_dot_fixedpt = (int16_t*)malloc(512*512*18 * sizeof(int16_t));
    atan2_snap_to_fixedpt(ori_best_bin_fixedpt, ori_dot_fixedpt);

    for(int dx=-255; dx <= 255; dx++){
        for(int dy=-255; dy <= 255; dy++)
        //int dy = 0;
        {
#if 0 //fixed-pt vs floating-pt
            if(ori_best_bin_floatpt[ (dx+255)*512 + (dy*255) ] != ori_best_bin_fixedpt[ (dx+255)*512 + (dy*255) ])
            {
                printf("mismatch: ori_best_bin_floatpt[dx=%d][dy=%d]=%d, ori_best_bin_fixedpt[dx=%d][dy=%d]=%d \n", dx, dy, ori_best_bin_floatpt[ (dx+255)*512 + (dy*255) ], dx, dy, ori_best_bin_fixedpt[ (dx+255)*512 + (dy*255) ]);
 
                for (int o = 0; o < 18; o++) {
//                    printf("    ori_dot_FLOATpt[dx=%d][dy=%d][%d] = %f \n", dx, dy, o, ori_dot_floatpt[(dx+255)*512*18 + (dy+255)*18 + o]); 
//                    printf("    ori_dot_FIXEDpt[dx=%d][dy=%d][%d] = %d \n", dx, dy, o, ori_dot_fixedpt[(dx+255)*512*18 + (dy+255)*18 + o]); 

                }
            }
#endif
#if 1 //floating-pt vs FFLD LUT
            if(ori_best_bin_floatpt[ (dx+255)*512 + (dy*255) ] != (ATAN2_TABLE[dy+255][dx+255]))
            {
                printf("    ori_best_bin_FLOATpt[dx=%d][dy=%d] = %d, ATAN2_TABLE[dy=%d][dx=%d] = %d \n", dx, dy, ori_best_bin_floatpt[ (dx+255)*512 + (dy*255) ], dy, dx, (ATAN2_TABLE[dy+255][dx+255])); 
            }

#endif
        }
    }
}

//print the lookup table, with stuff grouped by bin (look for patterns...)
void analyze_LUT(){

    int num_per_bin[19];

    for(int ori=0; ori<=18; ori++){
        num_per_bin[ori] = 0;

        printf("ori bin %d \n", ori);
        for(int dx=-255; dx <= 255; dx++){
            for(int dy=-255; dy <= 255; dy++){

                if(ATAN2_TABLE[dy+255][dx+255] == ori){
                    printf("    ATAN2_TABLE[dy=%d][dx=%d] = %d \n", dy, dx, ATAN2_TABLE[dy+255][dx+255]);
                    num_per_bin[ori]++;
                } 
            }
        }
    }

    //TODO: print num_per_bin[0:18]
    for(int ori=0; ori<=18; ori++){
        printf("%d LUT entries in bin %d \n", num_per_bin[ori], ori);
    }
}


int main (int argc, char **argv)
{
    fixedpt_vs_floatpt(); //compare voc5 floatpt, voc5 fixedpt, FFLD floatpt
    analyze_LUT(); //look for patterns/clusters in LUT

    return 0;
}
