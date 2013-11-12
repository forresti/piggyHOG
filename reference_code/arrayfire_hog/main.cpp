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

array gradient_builtin(array input){
    int width = input.dims(1);
    int height = input.dims(0);
    array gradX(height, width, 3, f32); //pre-allocate output
    array gradY(height, width, 3, f32); //I think even the input, by default, is f32.

    //af::grad() doesn't like multichannel images, so do 1 channel at a time 
    array gradX_1ch, gradY_1ch; //tmp results for 1 channel at a time
    for(int ch=0; ch<3; ch++){
        array input_1ch = input(span, span, ch);
        af::grad(gradX_1ch, gradY_1ch, input_1ch);
        gradX(span, span, ch) = gradX_1ch; //output is some goofy thing that sorta has shadows.
        gradY(span, span, ch) = gradY_1ch; //output looks ok
    }

    gradX = abs(gradX)*2; //do 'abs' so that img makes sense as [0 to 255]
    gradY = abs(gradY)*2;

    saveimage("gradX_builtin.jpg", gradX);
    saveimage("gradY_builtin.jpg", gradY);
    return gradY;
}

array gradient_gfor(array input){
    int width = input.dims(1);
    int height = input.dims(0);

    array gradX(height, width, 3, f32);
    array gradY(height, width, 3, f32); 

#if 0 //without gfor
    for(int x=0; x<width; x++){
        for(int y=0; y<height; y++){
            for(int ch=0; ch<3; ch++){
                gradX(y,x,ch) = input(y, CLAMP(x+1,0,width-1), ch) -
                                input(y, CLAMP(x-1,0,width-1), ch); //this is already type float
                gradY(y,x,ch) = input(CLAMP(y+1,0,height-1), x, ch) -
                                input(CLAMP(y-1,0,height-1), x, ch); 

            }
        }
    }
#endif
#if 0 //GPU, with gfor
    //gfor(array x, width){
    gfor(array y, height){
        //gradX(span, x, span) = input(span, CLAMP(x+1,0,width-1), span) -
        //                       input(span, CLAMP(x-1,0,width-1), span);
        //gradX(span, x, span) = input(span, CLAMP(x+1,0,width-1), span); //just shift right
        //gradX(span, x, span) = input(span, x, span); //middle dim in parallel ... just copy. 'unspecified launch failure'
        //aha, 'A(array, span, array) isn't supported. A(array, array, span) is supported." --http://forums.accelereyes.com/forums/viewtopic.php?f=17&t=6415
        gradX(y, span, span) = input(y, span, span); 
    }
#endif
#if 1 //trivial gfor example, with inner dim in parallel.
    gfor(array ch, 3){
        gradX(span, span, ch) = input(span, span, ch);
    }

#endif

//TODO: split array into its individual channels. 

    gradX = abs(gradX);
    gradY = abs(gradY);
    saveimage("gradX_gfor.jpg", gradX);
    saveimage("gradY_gfor.jpg", gradY);
    return gradX;
}

array bandwidth_gfor(array input){
    int width = input.dims(1);
    int height = input.dims(0);
    array input_copy(height, width, 3, f32); 
    af::sync();  
 
    double start_bwTest = read_timer();
    gfor(array ch, 3){
        //input_copy(span, span, ch) = input(span, span, ch); 
        input_copy(span, span, ch) = input(span, span, ch) * 0.5f;
    }
    //input_copy = input * 0.5f; //seems to be evaluated lazily ... takes no time.
    af::sync();
    double time_bwTest = read_timer() - start_bwTest;
    double gb_to_copy = width * height * 3 * sizeof(float) / 1e9;
    double gb_per_sec = gb_to_copy / (time_bwTest/1000); //convert time_bwTest from ms to sec

    printf("[gfor] stream benchmark in %f ms, %f gb/s \n", time_bwTest, gb_per_sec);
    return input_copy;
}

int main(int argc, char** argv) {
    //deviceset(1);
    try {
        info();
        //array input = loadimage("../../images_640x480/carsgraz_001.image.jpg", true); //iscolor='true'
#if 0
        array input = loadimage("Lena.jpg", true); //iscolor='true'
        printf("size of input: %d, %d, %d\n", input.dims(0), input.dims(1), input.dims(2));

    //warmup
        array dummy = gradient_builtin(input);
        af::sync();

    //builtin version
        double start_gradient = read_timer();
        array result_builtin = gradient_builtin(input);
        af::sync();
        double time_gradient = read_timer() - start_gradient;
        printf("[builtin] computed gradient in %f ms \n", time_gradient);
        //saveimage("./gradient_builtin.jpg", result_builtin);

    //gfor version
        start_gradient = read_timer();
        array result_gfor = gradient_gfor(input);
        af::sync();
        time_gradient = read_timer() - start_gradient;
        printf("[gfor] computed gradient in %f ms \n", time_gradient);
        //saveimage("./gradient_gfor.jpg", result_gfor);

    //bandwidth test
        int n_iter = 10;
        for(int i=0; i<n_iter; i++){
            array input_copy = bandwidth_gfor(input);
        }
#endif
    } catch (af::exception& e) {
        fprintf(stderr, "%s\n", e.what());
    }
    return 0;
}
