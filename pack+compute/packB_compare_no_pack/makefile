install:
	g++ -g pack_compute.cc -o pack_compute -lmkl_rt -O3 -mtune=znver1 -march=znver1

clean:
	rm -rf pack_compute

rmtxt:
	rm -rf pack_compute_Runtime.txt
	rm -rf pack_compute_Gflops.txt
