#Author: Forrest Iandola forresti@eecs.berkeley.edu

OBJS = main.o gradientMex.o helpers.o
EXENAME = main

CC = g++
CCOPTS = -c  -g -O0 `pkg-config opencv --cflags`
LINK = g++
LINKOPTS = `pkg-config opencv --libs`

all : $(EXENAME)
$(EXENAME) : $(OBJS)
	$(LINK) -o $(EXENAME) $(OBJS) $(LINKOPTS)

main.o : main.cpp helpers.h gradientMex.h
	$(CC) $(CCOPTS) main.cpp

helpers.o : helpers.cpp helpers.h
	$(CC) $(CCOPTS) helpers.cpp

gradientMex.o : gradientMex.cpp gradientMex.h
	$(CC) $(CCOPTS) gradientMex.cpp



clean : 
	rm -f *.o $(EXENAME) 2>/dev/null

