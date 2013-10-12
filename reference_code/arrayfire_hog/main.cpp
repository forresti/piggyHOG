// MODIFIED FROM ARRAYFIRE image_demo.cpp EXAMPLE PROGRAM
#include <stdio.h>
#include <arrayfire.h>
#include <sys/time.h>
#include <vector>
using namespace af;
using namespace std;

#define CLAMP(x, lo, hi)  max(lo, min(x, hi)) 

double read_timer(){
    struct timeval start;
    gettimeofday( &start, NULL );
    return (double)((start.tv_sec) + 1.0e-6 * (start.tv_usec)) * 1000; //in milliseconds
}

//for now, just return the orientations. (TODO: return the magnitudes too ... I may need to pass magnitude and orientation output arrays by reference)
array gradient_builtin(array input){
    int width = input.dims(1);
    int height = input.dims(0);

    array gradX(height, width, 3, f32); //I think even the input, by default, is f32.
    array gradY(height, width, 3, f32);

    array gradX_1ch, gradY_1ch; //tmp
    for(int ch=0; ch<3; ch++){
        array input_1ch = input(span, span, ch);
        grad(gradX_1ch, gradY_1ch, input_1ch);
        gradX(span, span, ch) = gradX_1ch; //output is some goofy thing that sorta has shadows.
        gradY(span, span, ch) = gradY_1ch;
    }

    //grad(gradX_ch0, gradY_ch0, input_ch0); //output doesn't look very good. (some sort of ugly shadow effect). 
    
    //return gradX;
    return gradY;
}

array gradient_gfor(array input){
    int width = input.dims(1);
    int height = input.dims(0);

    //array gradX, gradY;
    //array gradX_ch0(input(span, span, 0)); //using just 1 channel gives reasonable-looking output
    array gradX(height, width, 3, f32);

    //TODO: gfor loop
    for(int x=0; x<width; x++){
        for(int y=0; y<height; y++){
            for(int ch=0; ch<3; ch++){
                gradX(y,x,ch) = input(CLAMP(y,0,height-1), CLAMP(x+1,0,width-1), ch) - 
                                input(CLAMP(y,0,height-1), CLAMP(x-1,0,width-1), ch); //TODO: cast this stuff to float
            }
        }
    }

    return gradX;
}

int main(int argc, char** argv) {
    //deviceset(1);

    try {
        info();
        array input = loadimage("../../images_640x480/carsgraz_001.image.jpg", true); //iscolor='true'
        printf("size of input: %d, %d, %d\n", input.dims(0), input.dims(1), input.dims(2));

    //warmup
        array dummy = gradient_builtin(input);
        cudaDeviceSynchronize();

    //builtin version
        double start_gradient = read_timer();
        array result_builtin = gradient_builtin(input);
        cudaDeviceSynchronize();
        double time_gradient = read_timer() - start_gradient;
        printf("[builtin] computed gradient in %f ms \n", time_gradient);

        saveimage("./gradient_builtin.jpg", result_builtin);

#if 0
    //gfor version
        start_gradient = read_timer();
        array result_gfor = gradient_gfor(input);
        cudaDeviceSynchronize();
        time_gradient = read_timer() - start_gradient;
        printf("[gfor] computed gradient in %f ms \n", time_gradient);

        saveimage("./gradient_gfor.jpg", result_gfor);
#endif

    } catch (af::exception& e) {
        fprintf(stderr, "%s\n", e.what());
    }
    return 0;
}
