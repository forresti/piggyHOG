#include <emmintrin.h>
#include <math.h>
#include <sys/time.h>
#include <stdint.h>
#include <stdio.h>

#include "static_image.h"

//#define cimg_display 0
//#include "CImg.h"
//using namespace cimg_library;

timeval t1, t2;
#define begin_timing gettimeofday(&t1, NULL); for (int i = 0; i < 10; i++) {
#define end_timing } gettimeofday(&t2, NULL);

// typedef CImg<uint16_t> Image;

extern "C" {
#include "hog.h"
}

Image<uint16_t> blur_halide(Image<uint16_t> in) {
    Image<uint16_t> out(in.width()-8, in.height()-2);

    // Call it once to initialize the halide runtime stuff
//    halide_blur(in, out);

    begin_timing;

    // Compute the same region of the output as blur_fast (i.e., we're
    // still being sloppy with boundary conditions)
//    halide_blur(in, out);

    end_timing;

    return out;
}

int main(int argc, char **argv) {

    Image<uint16_t> input(6408, 4802);

    for (int y = 0; y < input.height(); y++) {
        for (int x = 0; x < input.width(); x++) {
            input(x, y) = rand() & 0xfff;
        }
    }

 //   Image<uint16_t> halide = blur_halide(input);
 //   float halide_time = (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000000.0f;

    // fast_time2 is always slower than fast_time, so skip printing it
//    printf("times: %f %f %f\n", slow_time, fast_time, halide_time);

#if 0
    for (int y = 64; y < input.height() - 64; y++) {
        for (int x = 64; x < input.width() - 64; x++) {
            if (blurry(x, y) != speedy(x, y) || blurry(x, y) != halide(x, y))
                printf("difference at (%d,%d): %d %d %d\n", x, y, blurry(x, y), speedy(x, y), halide(x, y));
        }
    }
#endif

    return 0;
}
