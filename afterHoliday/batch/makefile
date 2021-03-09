print.o:../print.cc
all:
	-g++  -c ../print.cc -o print.o
	-g++   -c cblas_sgemm_batch.cc  -o   batch.o
install:
	-g++ print.o batch.o  -lmkl_rt -O3 -Wall -march=znver1 -mtune=znver1 -o batch
	
clean:
	-rm batch print.o batch.o

rmrecord:
	-rm  recordGflops.log recordTime.log writeBatchGflops.log












#-lstdc++ -shared-libgcc
