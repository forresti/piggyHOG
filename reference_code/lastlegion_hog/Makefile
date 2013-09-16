#Author: Forrest Iandola forresti@eecs.berkeley.edu

OBJS = main.o  helpers.o

EXENAME = main

CC = g++
CCOPTS = -c  -g -pg -O0 `pkg-config opencv --cflags`
LINK = g++
LINKOPTS = -pg `pkg-config opencv --libs`

all : $(EXENAME)

$(EXENAME) : $(OBJS)
	$(LINK) -o $(EXENAME) $(OBJS) $(LINKOPTS)

main.o : main.cpp helpers.h hog.hpp
	$(CC) $(CCOPTS) main.cpp

helpers.o : helpers.cpp helpers.h
	$(CC) $(CCOPTS) helpers.cpp

clean : 
	rm -f *.o $(EXENAME) 2>/dev/null

