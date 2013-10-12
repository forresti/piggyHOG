#include <emmintrin.h>
#include <math.h>
#include <sys/time.h>
#include <stdint.h>
#include <stdio.h>
#include <string>

#include "static_image.h"
#include "image_io.h"
#include "common/helpers.cpp"

extern "C" {
#include "gradient.h"
}

int main(int argc, char **argv){
    std::string imgName = "../../images_640x480/carsgraz_001.image.png"; //TODO: get jpg support
    Image<uint8_t> input = load<uint8_t>(imgName.c_str()); //only supports png and ppm
    //Image<uint8_t> result(input.width(), input.height(), input.channels());
    Image<float> result(input.width(), input.height(), input.channels());
    gradient(input, result); 
    save(result, "./gradient.png");
    return 0;
}
