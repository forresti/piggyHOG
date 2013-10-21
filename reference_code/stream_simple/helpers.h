#ifndef __HELPERS_H__
#define __HELPERS_H__
#include <opencv2/opencv.hpp>
#include <sys/time.h>
#include <stdint.h> //for uintptr_t
#include <immintrin.h> //256-bit AVX
#include <xmmintrin.h> //for other SSE-like stuff

using namespace std;
using namespace cv;

//macros from voc-release5 fconvsse.cc
#define IS_ALIGNED(ptr) ((((uintptr_t)(ptr)) & 0xF) == 0) 

#if !defined(__APPLE__)
#include <malloc.h>
#define malloc_aligned(a,b) memalign(a,b)
#else
#define malloc_aligned(a,b) malloc(b)
#endif

double read_timer();
std::string forrestGetImgType(int imgTypeInt);

#endif

