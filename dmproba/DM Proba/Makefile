WARN = -Wmissing-prototypes -Wall #-Winline 
OPTI = -O3 -finline-functions -fomit-frame-pointer -DNDEBUG \
-fno-strict-aliasing --param max-inline-insns-single=1800 -march=native
STD = -std=c99
CC = gcc
SSE2FLAGS = -msse2 -DHAVE_SSE2
CCFLAGS = $(OPTI) $(WARN) $(STD) $(SSE2FLAGS)

dm: main.c ./dSFMT-src-2.1/dSFMT.c
	${CC} ${CCFLAGS} -DDSFMT_MEXP=521 -o $@ $^
	

