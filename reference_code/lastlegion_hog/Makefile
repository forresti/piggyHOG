#Author: Forrest Iandola iandola1@illinois.edu
#Created: 1/24/10
#Modified: 9/6/12

#
# Object files are listed here
#
OBJS = main.o  helpers.o

#
# Change your program name here
#
EXENAME = main

#
# Recognize these macro assignments (like g++), but leave them alone.
#
#CC = nvcc 
CC = g++
CCOPTS = -c  -g -pg -O0 `pkg-config opencv --cflags`
#LINK = nvcc
LINK = g++
LINKOPTS = -pg `pkg-config opencv --libs`

all : $(EXENAME)

#
# The main compiling step, using all macros.
#
$(EXENAME) : $(OBJS)
	$(LINK) -o $(EXENAME) $(OBJS) $(LINKOPTS)

#
# Here is where your work will be done
#
main.o : main.cpp helpers.h hog.hpp
	$(CC) $(CCOPTS) main.cpp

helpers.o : helpers.cpp helpers.h
	$(CC) $(CCOPTS) helpers.cpp

clean : 
	rm -f *.o $(EXENAME) 2>/dev/null

