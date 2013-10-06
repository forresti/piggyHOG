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



//argmax
    //RDom r(0,3);// < 3, or <= 3?
    RDom r(x,x+1, y,y+1, 0,3); //trying all 3 dims in RDom
    Func arg_max_f("arg_max_f");
    arg_max_f() = 0;
    //arg_max_f() = select( (mag_rgb(x,y,r) > mag_rgb(x,y,arg_max_f())), r, arg_max_f() );
    arg_max_f() = select( (mag_rgb(r.x, r.y, r.z) > mag_rgb(arg_max_f())), r, arg_max_f() );

//from 6/17/13 halide-dev mailing list
#if 0
Func f("f");
Func arg_max_f("arg_max_f");
f(x) = input(x,0,0);
RDom r(0, 100);
arg_max_f() = 0;
arg_max_f() = select(f(r) > f(arg_max_f()), r, arg_max_f());
#endif

    Func mag_argmax; //idx of channel with max gradient
    //mag_argmax(x, y) = mag_rgb(x, y, 0); //placeholder

    //gradX_rgb.compile_to_file("gradient", input, output); 
    //gradX_rgb.compile_to_file("gradient", input); //temporary
      
    mag_rgb.compile_to_file("gradient", input); //I want a 1-channel float output buffer...how do I do this?
  
//TODO: how I set things up so that 'output' is a separate space from 'input'?

    return 0;
}
