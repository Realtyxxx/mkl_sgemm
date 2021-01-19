install:
	g++ -g pack_compute.cc -o pack_compute -lmkl_rt -O3 -Wall

clean:
	rm -rf pack_compute

rmlog:
	rm  pack_compute_Runtime.log
	rm  pack_compute_Gflops.log
	rm  pack_compute_efficiency.log
	# rm  record.log
	# rm performance.log
	# script -a record.log
	# script -a performance.log
# trymulti:
# 	 export MKL_NUM_THREADS=1 && taskset -c 1  ./pack_compute 2000 320
# 	 export MKL_NUM_THREADS=2 && taskset -c 1  ./pack_compute 2000 320
# 	 export MKL_NUM_THREADS=3 && taskset -c 1  ./pack_compute 2000 320
# 	 export MKL_NUM_THREADS=4 && taskset -c 1  ./pack_compute 2000 320