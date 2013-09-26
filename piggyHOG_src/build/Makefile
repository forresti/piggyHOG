CXX = g++
CFLAGS = -g -fopenmp -O3 -fPIC -I/usr/local/include/ -I../eigen -I.. #-DFFLD_HOGPYRAMID_FELZENSZWALB_FEATURES  
LDLIBS = -g -lgomp -ljpeg 
OBJS = HOGPyramid.o JPEGImage.o helpers.o ffld.o
all:	ffld 

ffld: $(OBJS) 
	$(CXX) -o ffld $(OBJS) $(LDLIBS)

ffld.o: ../ffld.cpp ../HOGPyramid.h ../JPEGImage.h ../common/helpers.h
	$(CXX) $(CFLAGS) -c ../ffld.cpp

HOGPyramid.o:	../HOGPyramid.cpp ../HOGPyramid.h ../JPEGImage.h
	$(CXX) $(CFLAGS) -c ../HOGPyramid.cpp 

JPEGImage.o: ../JPEGImage.cpp ../JPEGImage.h
	$(CXX) $(CFLAGS) -c ../JPEGImage.cpp 

helpers.o: ../common/helpers.cpp ../common/helpers.h 
	$(CXX) $(CFLAGS) -c ../common/helpers.cpp 

clean:
	rm -f *~ *.o *.mex* *.obj 

