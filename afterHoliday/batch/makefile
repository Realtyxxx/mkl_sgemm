# print.o:../print.cc
# batch.o:cblas_sgemm_batch.cc


install: print.o batch.o
	-g++ print.o batch.o  -lmkl_rt -O3 -Wall -march=znver1 -mtune=znver1 -o batch

clean:
	-rm batch print.o batch.o

rmrecord:
	-rm   writeBatchGflops.log 


print.o:../print.cc
	-g++  -c ../print.cc -o print.o

batch.o:cblas_sgemm_batch.cc
	-g++   -c cblas_sgemm_batch.cc  -o   batch.o










#-lstdc++ -shared-libgcc
