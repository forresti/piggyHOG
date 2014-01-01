#Author: Forrest Iandola iandola1@illinois.edu
#Created: 1/24/10
#Modified: 9/6/12

#OBJS = main.o  helpers.o SimpleImg.o streamHog.o
OBJS = main.o helpers.o streamHog.o test_streamHog.o
EXENAME = main

#on sandy bridge, -march=native gives me -mavx -msse2avx
CC = g++
#CCOPTS = -c  -g -O3 `pkg-config opencv --cflags` -msse -ffast-math -fopenmp -mssse3 -mavx -msse2avx 
CCOPTS = -c  -g -O3 `pkg-config opencv --cflags` -msse -ffast-math -fopenmp -mssse3
LINK = g++
LINKOPTS = `pkg-config opencv --libs` -lgomp

all : $(EXENAME)

$(EXENAME) : $(OBJS)
	$(LINK) -o $(EXENAME) $(OBJS) $(LINKOPTS)

main.o : main.cpp helpers.h SimpleImg.hpp streamHog.h test_streamHog.h
	$(CC) $(CCOPTS) main.cpp

helpers.o : helpers.cpp helpers.h 
	$(CC) $(CCOPTS) helpers.cpp

#SimpleImg.o : SimpleImg.cpp SimpleImg.h helpers.h
#	$(CC) $(CCOPTS) SimpleImg.cpp

streamHog.o : streamHog.cpp streamHog.h helpers.h SimpleImg.hpp 
	$(CC) $(CCOPTS) streamHog.cpp

test_streamHog.o : test_streamHog.cpp test_streamHog.h helpers.h SimpleImg.hpp 
	$(CC) $(CCOPTS) test_streamHog.cpp

clean : 
	rm -f *.o $(EXENAME) 2>/dev/null

