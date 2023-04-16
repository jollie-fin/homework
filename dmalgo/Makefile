NB_NODES = 20
MACHINE_FILE = hosts
NBMACHINES = `cat hosts | sort | uniq | wc -l`
SRC = $(wildcard *.c)
PROGRAM = algopar
PARAM = /tmp/test

PWD= /home/prenaudg/mpich
CC = $(PWD)/bin/mpicc
CXX = $(PWD)/bin/mpic++
PEDANTIC_PARANOID_FREAK =       -g -O0 -Wall -Wshadow -Wcast-align \
				-Waggregate-return -Wstrict-prototypes \
				-Wredundant-decls -Wnested-externs \
				-Wpointer-arith -Wwrite-strings -finline-functions
REASONABLY_CAREFUL_DUDE =	-Wall
NO_PRAYER_FOR_THE_WICKED =	-w
WARNINGS = $(PEDANTIC_PARANOID_FREAK) 
CFLAGS = -g -O2 $(WARNINGS)

INCLUDES = 
DEFS =
LIBS = -lm -lpthread  -L$(PWD)/lib

C_FILES = $(wildcard *.c)
BIN_FILES = $(patsubst %.c,%,$(C_FILES))
O_FILES = $(patsubst %.c,%.o,$(C_FILES))

all: $(SRC:.c=)

run: algopar
	$(PWD)/bin/mpdboot -n $(NBMACHINES) -f $(MACHINE_FILE)
	time $(PWD)/bin/mpiexec -n $(NB_NODES) `pwd`/$(PROGRAM) $(PARAM)
	$(PWD)/bin/mpdallexit

%: %.o
	$(CXX) $(INCLUDES) $(DEFS) $(CFLAGS) $^ $(LIBS) $(LDADD) -o $@ 

%.o: %.c
	$(CC) $(INCLUDES) $(DEFS) $(CFLAGS) -c -o $@ $<

%.o: %.cpp
	$(CXX) $(INCLUDES) $(DEFS) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(BIN_FILES) *.o *~

.PHONY : clean
