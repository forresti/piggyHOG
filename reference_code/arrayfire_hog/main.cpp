// MODIFIED FROM ARRAYFIRE image_demo.cpp EXAMPLE PROGRAM
#include <stdio.h>
#include <arrayfire.h>
#include <sys/time.h>
#include <vector>
using namespace af;
using namespace std;

double read_timer()
{
    struct timeval start;
    gettimeofday( &start, NULL );
    return (double)((start.tv_sec) + 1.0e-6 * (start.tv_usec)); //in seconds
}

//for now, just return the orientations. (TODO: return the magnitudes too ... I may need to pass magnitude and orientation output arrays by reference)
array gradient(array input){

    array gradX, gradY;

    grad(gradX, gradY, input);

    return gradX;
}


int main(int argc, char** argv) {
    //cudaSetDevice(3);
    //deviceset(1);

    try {
        info();
        array input = loadimage("../../images_640x480/carsgraz_001.image.jpg");

        double start_gradient = read_timer();
        array result = gradient(input);
        double time_gradient = read_timer() - start_gradient;
        printf("computed gradient in %f ms \n", time_gradient);

        saveimage("./gradient.jpg", result);

    } catch (af::exception& e) {
        fprintf(stderr, "%s\n", e.what());
    }
    return 0;
}
