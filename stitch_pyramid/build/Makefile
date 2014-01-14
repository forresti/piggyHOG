CXX = g++
CFLAGS = -g -fopenmp -O3 -fPIC -I/usr/local/include/ -I../eigen -I.. #-DFFLD_HOGPYRAMID_FELZENSZWALB_FEATURES  
LDLIBS = -g -lgomp -ljpeg 
OBJS = JPEGPyramid.o JPEGImage.o helpers.o Patchwork.o Rectangle.o stitch_pyramid.o
all:	stitch_pyramid 

stitch_pyramid: $(OBJS) 
	$(CXX) -o stitch_pyramid $(OBJS) $(LDLIBS)

stitch_pyramid.o: ../stitch_pyramid.cpp ../JPEGPyramid.h ../JPEGImage.h ../Patchwork.h ../common/helpers.h
	$(CXX) $(CFLAGS) -c ../stitch_pyramid.cpp

JPEGPyramid.o:	../JPEGPyramid.cpp ../JPEGPyramid.h ../JPEGImage.h
	$(CXX) $(CFLAGS) -c ../JPEGPyramid.cpp 

JPEGImage.o: ../JPEGImage.cpp ../JPEGImage.h
	$(CXX) $(CFLAGS) -c ../JPEGImage.cpp 

helpers.o: ../common/helpers.cpp ../common/helpers.h 
	$(CXX) $(CFLAGS) -c ../common/helpers.cpp 

#TODO: rename 'Patchwork' to 'StitchPyramid'
Patchwork.o: ../Patchwork.cpp ../Patchwork.h ../JPEGPyramid.h ../Rectangle.h
	$(CXX) $(CFLAGS) -c ../Patchwork.cpp 

Rectangle.o: ../Rectangle.cpp ../Rectangle.h
	$(CXX) $(CFLAGS) -c ../Rectangle.cpp 

clean:
	rm -f *~ *.o *.mex* *.obj 

