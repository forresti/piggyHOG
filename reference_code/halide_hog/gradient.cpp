#include <Halide.h>
using namespace Halide;

int main(int argc, char **argv) {

    ImageParam input(UInt(8), 3);
//    ImageParam output(Float(32), 2);
    Func gradX_rgb("gradX_rgb"), gradY_rgb("gradY_rgb");
    Var x("x"), y("y"), ch("ch");
    Var xi("xi"), yi("yi");

    Func clamped("clamped");
    clamped(x,y,ch) = input(clamp(x,0,input.width()-1), clamp(y,0,input.height()-1), ch);

    //thanks for cast idea: github.com/halide/Halide/blob/master/tutorial/lesson_02.cpp    
    //Expr input_as_float = Halide::cast<float>(input);

    gradX_rgb(x, y, ch) = cast<float>(clamped(x+1, y, ch)) - cast<float>(clamped(x-1, y, ch));
    gradY_rgb(x, y, ch) = cast<float>(clamped(x, y+1, ch)) - cast<float>(clamped(x, y-1, ch));

    Func mag_rgb;
    mag_rgb(x, y, ch) = cast<uint8_t>(gradX_rgb(x,y,ch)*gradX_rgb(x,y,ch) + gradY_rgb(x,y,ch)*gradY_rgb(x,y,ch)); //uint8_t cast is temporary

    //gradX_rgb.compile_to_file("gradient", input, output); 
    //gradX_rgb.compile_to_file("gradient", input); //temporary
      
    mag_rgb.compile_to_file("gradient", input); //I want a 1-channel float output buffer...how do I do this?
  
//TODO: how I set things up so that 'output' is a separate space from 'input'?

    return 0;
}