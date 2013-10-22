#include "helpers.h"
#include "ForrestImg.h"
using namespace std;

ForrestImg::ForrestImg(int in_height, int in_width, int in_stride){
    height = in_height;
    width = in_width;
    stride = in_stride;

    //TODO: calloc?
    //TODO: align?
    //data = (PIXEL_TYPE*)malloc(width * stride * sizeof(PIXEL_TYPE)); 
    data = (PIXEL_TYPE*)malloc_aligned(32, width * stride * sizeof(PIXEL_TYPE));
}

ForrestImg::~ForrestImg(){
    free(data);
}

