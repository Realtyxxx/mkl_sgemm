install:
	rm -rf pack
	rm -rf nopack

	g++ nopack.cc -o nopack -lmkl_rt
	g++ pack.cc -o pack -lmkl_rt
	./nopack 512 500
	@echo -e  '\n'
	./pack 512 500

