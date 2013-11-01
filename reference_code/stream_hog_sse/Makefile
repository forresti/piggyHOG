#Author: Forrest Iandola iandola1@illinois.edu
#Created: 1/24/10
#Modified: 9/6/12

OBJS = main.o  helpers.o SimpleImg.o
EXENAME = main

#on sandy bridge, -march=native gives me -mavx -msse2avx
CC = g++
CCOPTS = -c  -g -O3 `pkg-config opencv --cflags` -march=native -ffast-math -fopenmp
LINK = g++
LINKOPTS = `pkg-config opencv --libs` -lgomp

all : $(EXENAME)

$(EXENAME) : $(OBJS)
	$(LINK) -o $(EXENAME) $(OBJS) $(LINKOPTS)

main.o : main.cpp helpers.h SimpleImg.h
	$(CC) $(CCOPTS) main.cpp

helpers.o : helpers.cpp helpers.h 
	$(CC) $(CCOPTS) helpers.cpp

SimpleImg.o : SimpleImg.cpp SimpleImg.h helpers.h
	$(CC) $(CCOPTS) SimpleImg.cpp

clean : 
	rm -f *.o $(EXENAME) 2>/dev/null

