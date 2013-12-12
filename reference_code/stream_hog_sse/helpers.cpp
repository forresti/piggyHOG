
#include "helpers.h"
using namespace std;

double read_timer(){
    struct timeval start;
    gettimeofday( &start, NULL );
    return (double)((start.tv_sec) + 1.0e-6 * (start.tv_usec)) * 1000; //in milliseconds
}

void print_epi16(__m128i vec_sse, string vec_name){
    int16_t vec_scalar[8];
    _mm_store_si128((__m128i*)vec_scalar, vec_sse);

    printf("    %s: ", vec_name.c_str());
    for(int i=0; i<8; i++){
        printf("%d, ", vec_scalar[i]);
    }
    printf("\n");
}

//size_per_element = sizeof(float), sizeof(char), or whatever data type you're using.
int compute_stride(int width, int size_per_element, int ALIGN_IN_BYTES){
    //thanks: http://stackoverflow.com/questions/2403631

    //int ALIGN_IN_BYTES=32; //for AVX
    int stride = (width*size_per_element + (ALIGN_IN_BYTES - (width*size_per_element)%ALIGN_IN_BYTES))/size_per_element;
    return stride;
}
