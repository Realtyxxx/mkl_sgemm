install: print.o pack.o
	-g++ print.o pack.o  -lmkl_rt -O3 -Wall -march=znver1 -mtune=znver1 -o pack

clean:
	-rm pack print.o pack.o

rmrecord:
	-rm  pack_compute_Gflops.log  pack_compute_Runtime.log


print.o:../print.cc
	-g++  -c ../print.cc -o print.o

pack.o:pack_compute.cc
	-g++   -c pack_compute.cc  -o   pack.o

