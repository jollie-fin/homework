NB_NODES = 5
SIZE_MATRIX = 24
MACHINE_FILE = hosts
NBMACHINES = `cat hosts | sort | uniq | wc -l`
SRC = $(wildcard *.c)
PROGRAM = matmult_template

CC = mpicc
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
LIBS = -lm -lpthread  -L/usr/lib

C_FILES = $(wildcard *.c)
BIN_FILES = $(patsubst %.c,%,$(C_FILES))
O_FILES = $(patsubst %.c,%.o,$(C_FILES))

all: $(SRC:.c=)

run: $(PROGRAM)
	mpiexec -n $(NB_NODES) `pwd`/$(PROGRAM) $(SIZE_MATRIX);

runcluster: $(PROGRAM)
	mpiexec -n $(NB_NODES) -hostfile hosts `pwd`/$(PROGRAM) $(SIZE_MATRIX);


%: %.o
	$(CC) $(INCLUDES) $(DEFS) $(CFLAGS) $^ $(LIBS) $(LDADD) -o $@ 

%.o: %.c
	$(CC) $(INCLUDES) $(DEFS) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(BIN_FILES) *.o *~

.SUFFIXES:
.PHONY : clean
