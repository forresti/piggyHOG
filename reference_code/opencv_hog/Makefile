#Author: Forrest Iandola iandola1@illinois.edu
#Created: 1/24/10
#Modified: 9/6/12

OBJS = main.o  helpers.o
EXENAME = main

CC = nvcc
CCOPTS = -c -g -pg -O0 `pkg-config opencv --cflags` #-I/opt/intel/ipp/include 
LINK = nvcc
LINKOPTS = -pg `pkg-config opencv --libs` #-L/opt/intel/ipp/lib/intel64 -L/opt/intel/composer_xe_2013.1.117/compiler/lib/intel64 -lippi -lipps -lippcore -liomp5 -lpthread -lm


all : $(EXENAME)

$(EXENAME) : $(OBJS)
	$(LINK) -o $(EXENAME) $(OBJS) $(LINKOPTS)

main.o : main.cpp helpers.h
	$(CC) $(CCOPTS) main.cpp

helpers.o : helpers.cpp helpers.h
	$(CC) $(CCOPTS) helpers.cpp

clean : 
	rm -f *.o $(EXENAME) 2>/dev/null

