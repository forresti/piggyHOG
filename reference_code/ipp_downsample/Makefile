OBJS = main.o  helpers.o

EXENAME = main

CC = g++
CCOPTS = -c -O3  -I/opt/intel/ipp/include
LINK = g++
LINKOPTS = -L/opt/intel/ipp/lib/intel64 -L/opt/intel/composer_xe_2013.1.117/compiler/lib/intel64 -lippi -lipps -lippcore -liomp5 -lpthread -lm

all : $(EXENAME)

$(EXENAME) : $(OBJS)
	$(LINK) -o $(EXENAME) $(OBJS) $(LINKOPTS)

main.o : main.cpp helpers.h
	$(CC) $(CCOPTS) main.cpp

helpers.o : helpers.cpp helpers.h
	$(CC) $(CCOPTS) helpers.cpp

clean : 
	rm -f *.o $(EXENAME) 2>/dev/null

