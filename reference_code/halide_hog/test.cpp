#include <emmintrin.h>
#include <math.h>
#include <sys/time.h>
#include <stdint.h>
#include <stdio.h>
#include <string>

#include "static_image.h"
#include "image_io.h"

timeval t1, t2;
#define begin_timing gettimeofday(&t1, NULL); for (int i = 0; i < 10; i++) {
#define end_timing } gettimeofday(&t2, NULL);

extern "C" {
#include "gradient.h"
}

Image<uint16_t> gradient_tester(Image<uint8_t> in) {
    Image<uint16_t> out(in.width(), in.height());

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

    //Image<uint16_t> input(6408, 4802);


    std::string imgName = "../../images_640x480/carsgraz_001.image.png"; //TODO: get jpg support 
    Image<uint8_t> input = load<uint8_t>(imgName.c_str()); //only supports png and ppm

    Image<uint16_t> halide = gradient_tester(input);
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
