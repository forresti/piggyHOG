#Author: Forrest Iandola forresti@eecs.berkeley.edu

OBJS = main.o gradientMex.o helpers_fhog.o helpers.o
EXENAME = main

CC = g++
CCOPTS = -c  -g -O0 `pkg-config opencv --cflags`
LINK = g++
LINKOPTS = -g `pkg-config opencv --libs`

all : $(EXENAME)
$(EXENAME) : $(OBJS)
	$(LINK) -o $(EXENAME) $(OBJS) $(LINKOPTS)

main.o : main.cpp helpers_fhog.h common/helpers.h gradientMex.h 
	$(CC) $(CCOPTS) main.cpp

helpers_fhog.o : helpers_fhog.cpp helpers_fhog.h
	$(CC) $(CCOPTS) helpers_fhog.cpp

helpers.o : common/helpers.cpp common/helpers.h
	$(CC) $(CCOPTS) common/helpers.cpp

gradientMex.o : gradientMex.cpp gradientMex.h
	$(CC) $(CCOPTS) gradientMex.cpp


clean : 
	rm -f *.o $(EXENAME) 2>/dev/null

