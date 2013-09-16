

## compile
gcc -o main main.c -I/opt/intel/ipp/include -L/opt/intel/ipp/lib/intel64 -L/opt/intel/composer_xe_2013.1.117/compiler/lib/intel64 -lippi -lipps -lippcore -liomp5 -lpthread -lm 

## run
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/ipp/lib/intel64 #for most libraries
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/composer_xe_2013.1.117/compiler/lib/intel64 #for iomp5
./main

## notes (for Forrest's machine, R8)
location of ipp.h: /opt/intel/composer_xe_2013.1.117/ipp/include/
location of iomp5 library: /opt/intel/composer_xe_2013.1.117/compiler/lib/intel64
location of other libraries: /opt/intel/ipp/lib/intel64


