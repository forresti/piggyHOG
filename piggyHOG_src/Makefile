OBJS = main.o  helpers.o PgHOG.o
EXENAME = main

CC = g++
CCOPTS = -c  -g -pg -O3 `pkg-config opencv --cflags`
LINK = g++
LINKOPTS = -pg `pkg-config opencv --libs`

all : $(EXENAME)
$(EXENAME) : $(OBJS)
	$(LINK) -o $(EXENAME) $(OBJS) $(LINKOPTS)

main.o : main.cpp PgHOG.h common/helpers.h 
	$(CC) $(CCOPTS) main.cpp

PgHOG.o : PgHOG.cpp PgHOG.h
	$(CC) $(CCOPTS) PgHOG.cpp

helpers.o: ../common/helpers.cpp ../common/helpers.h
	$(CXX) $(CFLAGS) -c ../common/helpers.cpp

clean : 
	rm -f *.o $(EXENAME) 2>/dev/null

