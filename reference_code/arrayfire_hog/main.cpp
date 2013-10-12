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

// 3x3 sobel weights
static const float h_sobel[] = {
    -2.0, -1.0,  0.0,
    -1.0,  0.0,  1.0,
     0.0,  1.0,  2.0
};

static double convolutionBenchmark(int kernelSize, array img_gray) 
{
    vector<float> hKernel(kernelSize*kernelSize, 1/float(kernelSize*kernelSize));
    array dKernel = array(kernelSize, kernelSize, &hKernel[0]);
    //array img_gray = loadimage("2250x2250.jpg");
    //img_gray = loadimage("../forrest_hacked_OpenCV_2_Cookbook_Code/9k_x_9k.png", false);

    double start = read_timer();
    array img_convolved = convolve(img_gray, dKernel); 
    cudaDeviceSynchronize();
    double responseTime = read_timer() - start;
    //saveimage("./Lena_convolved_ArrayFire.png", img_convolved);
    return responseTime;
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
