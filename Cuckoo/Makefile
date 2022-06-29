CC = gcc
FLAGS = -mavx2 -std=c++11 -march=haswell
DEBUG_FLAGS = 
EXEC = main
# EXEC = graph5 
# REG_OBJ = Heap.o QmaxO.o Qmax.o QmaxNew.o Skiplist.o
REG_OBJ = Cuckoo.o
OBJS = xxhash.o $(REG_OBJ) 




all: $(OBJS) $(EXEC)
	
clean:
	rm $(OBJS) $(EXEC)

xxhash.o: xxhash.c xxhash.h
	$(CC) -c $(DEBUG_FLAGS) $(FLAGS) $*.c

Cuckoo.o: Cuckoo.cpp Cuckoo.h
	$(CC) -c $(DEBUG_FLAGS) $(FLAGS) $*.cpp
	

