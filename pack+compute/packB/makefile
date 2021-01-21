install:
	g++ -g pack_compute.cc -o pack_compute -lmkl_rt -O3 -Wall -mtune=znver1 -march=znver1
clean:
	rm -rf pack_compute

rmtxt:
	rm -rf pack_compute_Runtime.txt
	rm -rf pack_compute_Gflops.txt
	rm -rf pack_compute_efficiency.txt

# trymulti:
# 	 export MKL_NUM_THREADS=1 && taskset -c 1  ./pack_compute 2000 320
# 	 export MKL_NUM_THREADS=2 && taskset -c 1  ./pack_compute 2000 320
# 	 export MKL_NUM_THREADS=3 && taskset -c 1  ./pack_compute 2000 320
# 	 export MKL_NUM_THREADS=4 && taskset -c 1  ./pack_compute 2000 320