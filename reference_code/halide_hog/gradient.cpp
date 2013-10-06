#include <Halide.h>
using namespace Halide;

int main(int argc, char **argv) {

    ImageParam input(UInt(8), 3);
//    ImageParam output(Float(32), 2);
    Func grad_x("grad_x"), grad_y("grad_y");
    Var x("x"), y("y"), ch("ch");
    Var xi("xi"), yi("yi");

    Func clamped("clamped");
    clamped(x,y,ch) = input(clamp(x,0,input.width()-1), clamp(y,0,input.height()-1), ch);

    //thanks for cast idea: github.com/halide/Halide/blob/master/tutorial/lesson_02.cpp    
    //Expr input_as_float = Halide::cast<float>(input);

#if 0
    // The algorithm
    grad_x(x, y) = (input(x, y) + input(x+1, y) + input(x+2, y))/3;
    grad_y(x, y) = (grad_x(x, y) + grad_x(x, y+1) + grad_x(x, y+2))/3;
    
    // How to schedule it
    grad_y.split(y, y, yi, 8).parallel(y).vectorize(x, 8);
    grad_x.store_at(grad_y, y).compute_at(grad_y, yi).vectorize(x, 8);  
    
    grad_y.compile_to_file("gradient", input); 
#endif

    grad_x(x, y, ch) = cast<uint8_t>(cast<float>(clamped(x+1, y, ch)) - cast<float>(clamped(x-1, y, ch)));
    //grad_x.compile_to_file("gradient", input, output); 
    grad_x.compile_to_file("gradient", input);
        
//TODO: how I set things up so that 'output' is a separate space from 'input'?

    return 0;
}
