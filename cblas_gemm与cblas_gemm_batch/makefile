install:
	g++ cblas_gemm.cc -o cblas_gemm -lmkl_rt
	g++ cblas_gemm_batch.cc -o cblas_gemm_batch -lmkl_rt

clean:
	rm cblas_gemm_batch
	rm cblas_gemm

rmrecord:
	rm  recordGflops.log
	rm  recordTime.log
	rm writeBatchGflops.log
	rm writeForGflops.log
