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

array cuComplex_to_float(array input)
{
    //assert(input.iscomplex() == true);
    //printf("iscomplex() = %d \n", input.iscomplex());
    int width=input.dims(0); 
    int height=input.dims(1);
    array output(width, height, f32);
    
    for(int y=0; y<height; y++)
    {
        for(int x=0; x<width; x++)
        {
            //FIXME
            //output(x,y)=input(x,y).r; //if this is really a cuComplex, it should have a .r real value
        }
    }

    return output;
}

double fftTest(array img_gray)
{
    int kernelSize=3;
    int paddedKernelWidth = img_gray.dims(0);
    int paddedKernelHeight = img_gray.dims(1);
    array dKernel = array(kernelSize, kernelSize, h_sobel);

    double start = read_timer();
    //array kernel_fft = fft2(dKernel, paddedKernelSize, paddedKernelSize);
    array kernel_fft = fft2(dKernel, paddedKernelWidth, paddedKernelHeight);
    array img_fft = fft2(img_gray);
    array img_kernel_fft = img_fft*kernel_fft; //element-wise multiply
    array img_convolved = ifft2(img_kernel_fft); //this is a cuComplex type
    cudaDeviceSynchronize();
    double responseTime = read_timer() - start;
    //array float_img_convolved = cuComplex_to_float(img_convolved);
    //array float_img_convolved = img_convolved.as(f32);
    //saveimage("./Lena_convolved_FFT_ArrayFire.png", float_img_convolved);
    return responseTime;
}

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

int main(int argc, char** argv) {
    //cudaSetDevice(3);
    //deviceset(1);

    try {
        info();
        //array img_gray = loadimage("../forrest_hacked_OpenCV_2_Cookbook_Code/9k_x_9k.png", false);
        //array img_gray = loadimage("../benchmarking_codeGen_conv/Lena.pgm", false);
        array img_gray = loadimage("./1920x1080.jpg");
//        convolutionBenchmark(3, img_gray); //warmup
        int nRuns = 10;
        double execTime = 0;
        int imgCols = img_gray.dims(0); int imgRows = img_gray.dims(1);
        for(int i=0; i<nRuns; i++)
        {
            execTime += fftTest(img_gray);
        }
        printf("FFT: imgSize = %dx%d, avg execTime = %f \n", imgCols, imgRows, execTime/nRuns);
        for(int kernelSize=2; kernelSize<9; kernelSize++)
        {
            execTime = 0;
            for(int i=0; i<nRuns; i++)
            {
                execTime += convolutionBenchmark(kernelSize, img_gray);
            }
            printf("imgSize = %dx%d,  kernelSize = %d,  avg execTime = %f \n", imgCols, imgRows, kernelSize, execTime/nRuns);
        }
    } catch (af::exception& e) {
        fprintf(stderr, "%s\n", e.what());
    }
    return 0;
}
