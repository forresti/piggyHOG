#include "helpers.h"
#include "ForrestImg.h"
using namespace std;

ForrestImg::ForrestImg(int in_height, int in_width, int in_stride){
    height = in_height;
    width = in_width;
    stride = in_stride;

    //TODO: calloc?
    //TODO: align?
    //data = (pixel_t*)malloc(width * stride * sizeof(pixel_t)); 
    data = (pixel_t*)malloc_aligned(32, width * stride * sizeof(pixel_t));
}

ForrestImg::~ForrestImg(){
    free(data);
}

