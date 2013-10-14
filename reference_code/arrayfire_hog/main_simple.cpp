// MODIFIED FROM ARRAYFIRE image_demo.cpp EXAMPLE PROGRAM
#include <stdio.h>
#include <arrayfire.h>
#include <sys/time.h>
#include <vector>
using namespace af;
using namespace std;

void gradient_builtin(array input){
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
 
    saveimage("gradX_arrayfire.jpg", gradX);
    saveimage("gradY_arrayfire.jpg", gradY);
}

int main(int argc, char** argv) {
    try {
        info();
        //array input = loadimage("../../images_640x480/carsgraz_001.image.jpg", true); //iscolor='true'
        array input = loadimage("./Lena.jpg", true); //iscolor='true'
        //array input = loadimage("../Lena.jpg", true); //iscolor='true'
        printf("size of input: %d, %d, %d\n", input.dims(0), input.dims(1), input.dims(2));

        gradient_builtin(input);

    } catch (af::exception& e) {
        fprintf(stderr, "%s\n", e.what());
    }
    return 0;
}
