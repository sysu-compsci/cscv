MKL_LD = -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread
TBB_LD = -ltbb

CFLAGS    = -O3 -march=native -g -fopenmp
CXXFLAGS  = -O3 -march=native -g -fopenmp -std=c++17 -fno-strict-aliasing
LDFLAGS   = -O3 -march=native -g -fopenmp ${MKL_LD} ${TBB_LD}

CC  = gcc
CXX = g++

# CC = icc
# CXX = icpc

# CFLAGS    = -O3 -xHost -mkl -ltbb -qopenmp -std=c++17 -g
# CXXFLAGS  = -O3 -xHost -mkl -ltbb -qopenmp -std=c++17 -g
# LDFLAGS   = -O3 -xHost -mkl -ltbb -qopenmp -std=c++17 -g

U00_EXENAME = u00-exe

INCLUDE = -I.

COMMON_CSRCS     = $(wildcard *.c */*.c)
COMMON_CPPSRCS   = $(wildcard *.cc */*.cc)
COMMON_OBJS      = $(COMMON_CPPSRCS:.cc=.o) $(COMMON_CSRCS:.c=.o)

U00_CPPSRCS    = $(wildcard unit-test/00/*.cc)
U00_CSRCS      = $(wildcard unit-test/00/*.c)
U00_OBJS	   = $(U00_CPPSRCS:.cc=.o) $(U00_CSRCS:.c=.o)

u00: $(U00_EXENAME)

all: u00

$(U00_EXENAME): $(COMMON_OBJS) $(U00_OBJS)
	$(CXX) $(LDFLAGS) -o $(U00_EXENAME) $^

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -o $@

%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDE) -c $< -o $@

clean:
	$(RM) *.o */*.o */*/o */*/*/*.o
	$(RM) $(U00_EXENAME)

clear:
	make clean
