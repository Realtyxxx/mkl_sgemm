// #include <sys/time.h>
// #include <time.h>
#include <fstream>
#include <iostream>
#include <vector>
#include "mkl.h"
#include "print.hpp"

using namespace std;

int main(int argc, char **argv) {
  MKL_INT Arg_G_Size = atoi(argv[1]);
  MKL_INT Arg_MKN = atoi(argv[2]);

  MKL_INT groupSize = Arg_G_Size;
  int m, k, n;
  m = k = n = Arg_MKN;

  float alpha = 1.0, beta = 0.0;

  float *a = (float *)malloc((size_t)sizeof(float) * k * n * groupSize);
  float *b = (float *)malloc((size_t)sizeof(float) * k * n);
  float *c = (float *)malloc((size_t)sizeof(float) * k * n * groupSize);

  // size_t : unsigned long
  // size_t size = cblas_sgemm_pack_get_size(CblasBMatrix, m, n, k);
  //float *Bp = (float *)mkl_malloc(size, 64);
  // CBLA_IDENTIFIER:Specifies which matrix is to be packed:
  // If identifier = CblasAMatrix, the size returned is the size required to
  // store matrix A in an internal format.
  // If identifier = CblasBMatrix, the sizereturned is the size required to
  // store matrix B in an internal format.

  int aSize = k * n * groupSize;
  int bSize = k * n;

  // vector<float> a(k * n * groupSize, 1);
  // vector<float> b(m * k, 2);
  // vector<float> c(k * n * groupSize);
  srand((unsigned)time(0));  // set the time seed;

  for (int i = 0; i < aSize; i++) a[i] = rand() / (float)(RAND_MAX / 9999);
  for (int i = 0; i < bSize; i++) b[i] = rand() / (float)(RAND_MAX / 9999);
  // for (int i = 0; i < aSize; i++) a[i] = 1;
  // for (int i = 0; i < bSize; i++) b[i] = 1;

  float *a_array[groupSize], *c_array[groupSize];

  for (int i = 0; i < groupSize; i++) {
    a_array[i] = a + k * n * i;
    c_array[i] = c + k * n * i;
  }
  double initial, end;
  initial = dsecnd();
  
  for (int i = 0; i < groupSize; i++) {
    cblas_sgemm_compute(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k,
                        a_array[i], k, b, n, beta, c_array[i], n);
  }
  end = dsecnd();

  double elapsed = end - initial;

  // printVector(a, m);
  // printVector(b, m);
  // printVector(c, m, groupSize);

  printf(
      " == Multiple Matrix multiplication (groupsize = %d, m n k = %d )using "
      "Intel(R) MKL cblas_sgemm "
      "completed == \n"
      " == at %.5f milliseconds == \n\n",
      Arg_G_Size, Arg_MKN, (elapsed * 1000));

  double sgemm_gflops = (2.0 * ((double)n) * ((double)m) * ((double)k) *
                         ((double)Arg_G_Size) * 1e-9);
  free(a);
  free(b);
  free(c);


  ofstream writeGflops, writeRuntime;
  writeGflops.open("pack_compute_Gflops.txt", ios::app);
  writeRuntime.open("pack_compute_Runtime.txt", ios::app);

  writeGflops << sgemm_gflops / elapsed << "    ";
  writeRuntime << elapsed * 1000 << "    ";

  writeRuntime.close();
  writeGflops.close();
}

// 就是频率 向量长度 2 核数 乘起来
// 这个是计算sgemm的计算量,这个值除以理论峰值就是效率
