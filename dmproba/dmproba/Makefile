# @file  Makefile
# @brief Makefile
#
# @author Mutsuo Saito (Hiroshima University)
# @author Makoto Matsumoto (Hiroshima University)
#
# Copyright (C) 2009, 2010 Mutsuo Saito, Makoto Matsumoto and
# Hiroshima University. All rights reserved.
#
# The new BSD License is applied to this software.
# see LICENSE.txt
#

#DEBUG = -DDEBUG -g -O0
WARN = -Wmissing-prototypes -Wall #-Winline
#OPTI = -O3 -finline-functions -fomit-frame-pointer -DNDEBUG -fno-strict-aliasing
OPTI = -g

STD = -std=c99
CC = gcc
#ARCH = -arch=compute_11
ARCH = -arch=compute_13
NVCC = /usr/local/cuda/bin/nvcc
NVCCARGS = $(ARCH) --compiler-options '-fopenmp -march=native -Wall -O3 -finline-functions -fomit-frame-pointer -fno-strict-aliasing'

# for OS X
#CUTILINC = -I"/Developer/GPU Computing/C/common/inc" -I../
#CUTILLIB = -L"/Developer/GPU Computing/C/lib" -lcutil_i386
# for linux
CUTILINC = -I$(HOME)/NVIDIA_GPU_Computing_SDK/C/common/inc\
		 -I$(HOME)/MTCUDA/MTGP-src-1.0.2/ #-I/usr/include/linux -I/usr/local/cuda/include 
CUTILLIB = -L/usr/lib/nvidia-current/ -L$(HOME)/NVIDIA_GPU_Computing_SDK/C/lib -lcutil_x86_64

CUDALINK = -lcuda
CPP = g++
CPPFLAGS = -Wall -Wextra -O3
CCFLAGS = $(OPTI) $(WARN) $(STD) $(DEBUG)

.SUFFIXES:	.cu .cubin .ptx .bin

all:jeudelavie
	./jeudelavie

run:exercice1 exercice2
	./exercice1
	./exercice2


exercice1: exercice1.cu mtgp32-fast.h mtgp32dc-param-11213.c \
	mtgp32-fast.o mtgp32-cuda-common.c mtgp-cuda-common.c
	${NVCC} ${NVCCARGS} -o $@ exercice1.cu mtgp32-fast.o \
	 ${CUTILINC} ${CUTILLIB} ${CUDALINK}

exercice2: exercice2.cu mtgp32-fast.h mtgp32dc-param-11213.c \
	mtgp32-fast.o mtgp32-cuda-common.c mtgp-cuda-common.c
	${NVCC} ${NVCCARGS} -o $@ exercice2.cu mtgp32-fast.o \
	 ${CUTILINC} ${CUTILLIB} ${CUDALINK}

jeudelavie: jeudelavie.cu Makefile
	${NVCC} ${NVCCARGS} -o $@ jeudelavie.cu \
	 ${CUTILINC} ${CUTILLIB} ${CUDALINK}

			    
.c.o:
	${CC} ${CCFLAGS} -c $<

clean:
	rm -f *.o *~



