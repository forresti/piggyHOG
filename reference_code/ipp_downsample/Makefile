#Author: Forrest Iandola iandola1@illinois.edu
#Created: 1/24/10
#Modified: 9/6/12

##Need to add the following to LD_LIBRARY_PATH in order to run:
#LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/ipp/lib/intel64 #for most libraries
#LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/composer_xe_2013.1.117/compiler/lib/intel64 #for iomp5

OBJS = main.o helpers_ipp.o helpers.o
EXENAME = main

CC = g++
CCOPTS = -c -fpermissive -g -pg -O0 `pkg-config opencv --cflags` -I/opt/intel/ipp/include 
LINK = g++
LINKOPTS = -pg `pkg-config opencv --libs` -L/opt/intel/ipp/lib/intel64 -L/opt/intel/composer_xe_2013.1.117/compiler/lib/intel64 -lippi -lipps -lippcore -liomp5 -lpthread -lm


all : $(EXENAME)

$(EXENAME) : $(OBJS)
	$(LINK) -o $(EXENAME) $(OBJS) $(LINKOPTS)

main.o : main.cpp helpers_ipp.h common/helpers.h
	$(CC) $(CCOPTS) main.cpp

helpers_ipp.o : helpers_ipp.cpp helpers_ipp.h    
	$(CC) $(CCOPTS) helpers_ipp.cpp

helpers.o : common/helpers.cpp common/helpers.h
	$(CC) $(CCOPTS) common/helpers.cpp

clean : 
	rm -f *.o $(EXENAME) 2>/dev/null

