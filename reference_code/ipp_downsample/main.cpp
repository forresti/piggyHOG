//from here: http://software.intel.com/en-us/articles/how-to-build-ipp-application-in-linux-environment/
#include "ipp.h"
int main(int argc, char **argv)
{
    IppiSize roi = {5,4};
    Ipp8u x[8*4] = {0};
    ippiSet_8u_C1R(1, x,8, roi); //call ippiSet_8u_C1R() function in ipp.h

    return 0;
}


