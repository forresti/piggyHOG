#include <Halide.h>
using namespace Halide;

int main(int argc, char **argv) {

    ImageParam input(UInt(8), 3);
    //ImageParam output(Float(32), 2);
    //OutputImageParam output(Float(32), 2); //'protected' in Halide.h
    Func gradX_rgb("gradX_rgb"), gradY_rgb("gradY_rgb");
    Var x("x"), y("y"), ch("ch");
    Var xi("xi"), yi("yi");

    Image<float> tmpImg;

    Func clamped("clamped");
    clamped(x,y,ch) = input(clamp(x,0,input.width()-1), clamp(y,0,input.height()-1), ch);

    //thanks for cast idea: github.com/halide/Halide/blob/master/tutorial/lesson_02.cpp    
    //Expr input_as_float = Halide::cast<float>(input);

    gradX_rgb(x, y, ch) = cast<float>(clamped(x+1, y, ch)) - cast<float>(clamped(x-1, y, ch));
    gradY_rgb(x, y, ch) = cast<float>(clamped(x, y+1, ch)) - cast<float>(clamped(x, y-1, ch));

    Func mag_rgb;
    mag_rgb(x, y, ch) = gradX_rgb(x,y,ch)*gradX_rgb(x,y,ch) + gradY_rgb(x,y,ch)*gradY_rgb(x,y,ch); //uint8_t cast is temporary

    Func mag_argmax; //idx of channel with max gradient
    //mag_argmax(x, y) = mag_rgb(x, y, 0); //placeholder
    mag_argmax(x, y) = select( (mag_rgb(x,y,0)>mag_rgb(x,y,1) && mag_rgb(x,y,0)>mag_rgb(x,y,2)), 0,  //argmax=0 if ch0>ch1 and ch0>ch2 
                               select( (mag_rgb(x,y,1)>mag_rgb(x,y,0) && mag_rgb(x,y,1)>mag_rgb(x,y,2)), 1, 2) ); //argmax=1 if ch1>ch0 and ch2>ch0, else argmax=2

    Func mag;
    mag(x, y) = mag_rgb(x, y, mag_argmax(x,y)); //this is expensive for some reason...
    
    Func ori;
    //ori(x, y) = atan2( cast<double>(gradY_rgb(x, y, mag_argmax(x,y))), cast<double>(gradX_rgb(x, y, mag_argmax(x,y))) );
    ori(x,y) = Halide::atan2(mag(x,y), mag(x,y)); //dummy

    mag.compile_to_file("gradient", input);   


//TODO: how I set things up so that 'output' is a separate space from 'input'?

    return 0;
}

//old stuff -- argmax tests
//from 6/17/13 halide-dev mailing list -- compiles
#if 0
Func f("f");
Func arg_max_f("arg_max_f");
f(x) = input(x,0,0);
RDom r(0, 100);
arg_max_f() = 0;
arg_max_f() = select(f(r) > f(arg_max_f()), r, arg_max_f());
#endif



