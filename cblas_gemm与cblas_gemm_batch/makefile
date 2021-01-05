install:
	g++ cblas_gemm_batch.cc -o cblas_gemm_batch -lmkl_rt

	g++ cblas_gemm.cc -o cblas_gemm -lmkl_rt

clean:
	rm cblas_gemm_batch
	rm cblas_gemm



rmtxt:
	rm -rf recordGflops.txt
	rm -rf recordTime.txt
	rm writeBatchGflops.txt
	rm writeForGflops.txt