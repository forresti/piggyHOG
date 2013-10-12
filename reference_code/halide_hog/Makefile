
#Forrest Iandola
#   based on halide_blur provided Makefile
#   this is edited from a version of Halide pulled on 10/4/13

GXX ?= g++
HALIDE_DIR=~/installers/halide_git/Halide

LIBPNG_LIBS_DEFAULT = $(shell libpng-config --ldflags)
LIBPNG_CXX_FLAGS ?= $(shell libpng-config --cflags)
# Workaround for libpng-config pointing to 64-bit versions on linux even when we're building for 32-bit
ifneq (,$(findstring -m32,$(CXX)))
ifneq (,$(findstring x86_64,$(LIBPNG_LIBS_DEFAULT)))
LIBPNG_LIBS ?= -lpng
endif
endif
LIBPNG_LIBS ?= $(LIBPNG_LIBS_DEFAULT)
PNGFLAGS=$(LIBPNG_LIBS) $(LIBPNG_CXX_FLAGS)

ifeq ($(HL_TARGET),ptx)
  CUDA_LDFLAGS=-L/usr/lib/nvidia-current -lcuda
endif
ifeq ($(HL_TARGET),ptx-debug)
  CUDA_LDFLAGS=-L/usr/lib/nvidia-current -lcuda
endif

#TODO: find a way to automatically build gradient.o too
all: gradient test 

gradient: $(HALIDE_DIR) gradient.cpp
	$(MAKE) -C $(HALIDE_DIR)/ bin/libHalide.a
	$(GXX) gradient.cpp -I $(HALIDE_DIR)/include $(HALIDE_DIR)/bin/libHalide.a -o gradient -ldl -lpthread

gradient.o: gradient
	./gradient

# -O2 is faster than -O3 for this app (O3 unrolls too much)
test: test.cpp gradient.o
	$(GXX) -msse2 -fopenmp -Wall -O2 -I $(HALIDE_DIR)/apps/support/ test.cpp gradient.o -o test -lpthread -ldl $(PNGFLAGS) $(CUDA_LDFLAGS)

clean:
	rm -f test gradient.o gradient
