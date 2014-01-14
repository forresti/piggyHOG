CXX = g++
CFLAGS = -g -fopenmp -O3 -fPIC -I/usr/local/include/ -I../eigen -I.. #-DFFLD_HOGPYRAMID_FELZENSZWALB_FEATURES  
LDLIBS = -g -lgomp -ljpeg 
OBJS = HOGPyramid.o JPEGImage.o helpers.o Patchwork.o Rectangle.o ffld.o
all:	ffld 

ffld: $(OBJS) 
	$(CXX) -o ffld $(OBJS) $(LDLIBS)

#TODO: rename 'ffld' to 'main'
ffld.o: ../ffld.cpp ../HOGPyramid.h ../JPEGImage.h ../Patchwork.h ../common/helpers.h
	$(CXX) $(CFLAGS) -c ../ffld.cpp

HOGPyramid.o:	../HOGPyramid.cpp ../HOGPyramid.h ../JPEGImage.h
	$(CXX) $(CFLAGS) -c ../HOGPyramid.cpp 

JPEGImage.o: ../JPEGImage.cpp ../JPEGImage.h
	$(CXX) $(CFLAGS) -c ../JPEGImage.cpp 

helpers.o: ../common/helpers.cpp ../common/helpers.h 
	$(CXX) $(CFLAGS) -c ../common/helpers.cpp 

#TODO: rename 'Patchwork' to 'StitchPyramid'
Patchwork.o: ../Patchwork.cpp ../Patchwork.h ../HOGPyramid.h ../Rectangle.h
	$(CXX) $(CFLAGS) -c ../Patchwork.cpp 

Rectangle.o: ../Rectangle.cpp ../Rectangle.h
	$(CXX) $(CFLAGS) -c ../Rectangle.cpp 

clean:
	rm -f *~ *.o *.mex* *.obj 

