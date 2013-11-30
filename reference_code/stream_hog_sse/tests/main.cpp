
//#include <cuda.h>
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


int main (int argc, char **argv)
{

    return 0;
}
