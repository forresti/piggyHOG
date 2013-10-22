#Author: Forrest Iandola iandola1@illinois.edu
#Created: 1/24/10
#Modified: 9/6/12

OBJS = main.o  helpers.o ForrestImg.o
EXENAME = main

CC = g++
CCOPTS = -c  -g -O3 -fopenmp -mavx -msse2avx
LINK = g++
LINKOPTS = -lgomp 

all : $(EXENAME)

$(EXENAME) : $(OBJS)
	$(LINK) -o $(EXENAME) $(OBJS) $(LINKOPTS)

main.o : main.cpp helpers.h ForrestImg.h
	$(CC) $(CCOPTS) main.cpp

helpers.o : helpers.cpp helpers.h 
	$(CC) $(CCOPTS) helpers.cpp

ForrestImg.o : ForrestImg.cpp ForrestImg.h helpers.h
	$(CC) $(CCOPTS) ForrestImg.cpp

clean : 
	rm -f *.o $(EXENAME) 2>/dev/null

