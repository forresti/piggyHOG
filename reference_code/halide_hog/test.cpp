#include <emmintrin.h>
#include <math.h>
#include <sys/time.h>
#include <stdint.h>
#include <stdio.h>
#include <string>

#include "static_image.h"
#include "image_io.h"

extern "C" {
#include "gradient.h"
}

double read_timer(){
    struct timeval start;
    gettimeofday( &start, NULL );
    return (double)((start.tv_sec) + 1.0e-6 * (start.tv_usec)) * 1000; //in milliseconds
}

int main(int argc, char **argv){
    std::string imgName = "../../images_640x480/carsgraz_001.image.png"; //TODO: get jpg support
    Image<uint8_t> input = load<uint8_t>(imgName.c_str()); //only supports png and ppm
    //Image<uint8_t> result(input.width(), input.height(), input.channels());
    Image<float> result(input.width(), input.height(), 1);

    double start_gradient = read_timer();
    gradient(input, result); 
    double time_gradient = read_timer() - start_gradient;
    printf("computed gradient in %f ms \n", time_gradient);

    save(result, "./gradient.png");
    return 0;
}
