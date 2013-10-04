OBJS = main.o  helpers.o PgHog.o
EXENAME = main

CC = g++
CCOPTS = -c -g -fopenmp -O3 `pkg-config opencv --cflags` -funroll-all-loops
LINK = g++
LINKOPTS = -g -lgomp `pkg-config opencv --libs`

all : $(EXENAME)
$(EXENAME) : $(OBJS)
	$(LINK) -o $(EXENAME) $(OBJS) $(LINKOPTS)

main.o : main.cpp PgHog.h common/helpers.h 
	$(CC) $(CCOPTS) main.cpp

PgHog.o : PgHog.cpp PgHog.h PgHogContainer.h common/helpers.h
	$(CC) $(CCOPTS) PgHog.cpp

helpers.o: common/helpers.cpp common/helpers.h
	$(CXX) $(CFLAGS) -c common/helpers.cpp

clean : 
	rm -f *.o $(EXENAME) 2>/dev/null

