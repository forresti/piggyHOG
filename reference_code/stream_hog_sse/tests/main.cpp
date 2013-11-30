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

//stuff for approximate vectorized atan2
void init_atan2_constants(){
    for(int i=0; i<9; i++){
        uu_fixedpt[i] = round(uu[i] * 100);
        vv_fixedpt[i] = round(vv[i] * 100);

        //vector of copies of uu and vv for SSE vectorization
        uu_fixedpt_epi16[i] = _mm_set_epi16(uu_fixedpt[i],uu_fixedpt[i],uu_fixedpt[i],uu_fixedpt[i],uu_fixedpt[i],uu_fixedpt[i],uu_fixedpt[i],uu_fixedpt[i]);
        vv_fixedpt_epi16[i] = _mm_set_epi16(vv_fixedpt[i],vv_fixedpt[i],vv_fixedpt[i],vv_fixedpt[i],vv_fixedpt[i],vv_fixedpt[i],vv_fixedpt[i],vv_fixedpt[i]);
    }
}

//@in-out ori_best_bin[dx][dy] = atan2 ori bin that best matches the gradient (dx,dy) 
//@in-out ori_dot[dx][dy][bin] = how well we match with each orientation bin
//uses FIXED-point calculations. (this is the reference implementation from voc-release5)
void atan2_snap_to_fixedpt(int* ori_best_bin, int16_t* ori_dot){

    for(int16_t dx=-255; dx <= 255; dx++){
        for(int16_t dy=-255; dy <= 255; dy++){

            int16_t best_dot = 0;
            int best_o = 0;

            for (int o = 0; o < 9; o++) {
                //double dot = uu[o]*dx + vv[o]*dy; //float
                uint16_t dot = uu_fixedpt[o]*dx + vv_fixedpt[o]*dy;

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
                double dot = uu[o]*dx + vv[o]*dy; //float
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

int main (int argc, char **argv)
{

  //floating-pt "best ori bin" experiment
    int* ori_best_bin_floatpt = (int*)malloc(512*512 * sizeof(int));
    float* ori_dot_floatpt = (float*)malloc(512*512*18 * sizeof(float));
    atan2_snap_to_floatpt(ori_best_bin_floatpt, ori_dot_floatpt);

  //fixed-pt "best ori bin" experiment
    int* ori_best_bin_fixedpt = (int*)malloc(512*512 * sizeof(int));
    int16_t* ori_dot_fixedpt = (int16_t*)malloc(512*512*18 * sizeof(int16_t));
    atan2_snap_to_fixedpt(ori_best_bin_fixedpt, ori_dot_fixedpt);

    for(int dx=-255; dx <= 255; dx++){
        for(int dy=-255; dy <= 255; dy++){
            for (int o = 0; o < 18; o++) {

                
                printf(    "ori_dot_FLOATpt[%d][%d][%d] = %f \n", dx, dy, o, ori_dot_floatpt[(dx+255)*512*18 + (dy+255)*18 + o]); 
                printf(    "ori_dot_FIXEDpt[%d][%d][%d] = %d \n", dx, dy, o, ori_dot_fixedpt[(dx+255)*512*18 + (dy+255)*18 + o]); 

            }
        }
    }

    return 0;
}
