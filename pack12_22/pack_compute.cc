// #include <sys/time.h>
// #include <time.h>
#include <fstream>
#include <iostream>
#include <vector>
#include "mkl.h"

using namespace std;

// static struct timeval start;
// static struct timeval thisend;
// void tic(void) { gettimeofday(&start, NULL); }
// double toc(void) {
//   gettimeofday(&thisend, NULL);
//   return (thisend.tv_sec - start.tv_sec) +
//          1.0e-6 * (thisend.tv_usec - start.tv_usec);
// }

int main(int argc, char** argv) {
  srand((unsigned)time(0));  // set the time seed;

  MKL_INT Arg_G_Size = atoi(argv[1]);
  MKL_INT Arg_MKN = atoi(argv[2]);

  MKL_INT groupSize = Arg_G_Size;
  int m, k, n;
  m = k = n = Arg_MKN;
  // size_t : unsigned long
  size_t size = cblas_sgemm_pack_get_size(CblasAMatrix, m, n, k);
  // CBLA_IDENTIFIER:Specifies which matrix is to be packed:
  // If identifier = CblasAMatrix, the size returned is the size required to
  // store matrix A in an internal format.
  // If identifier = CblasBMatrix, the sizereturned is the size required to
  // store matrix B in an internal format.
  float* Ap = (float*)mkl_malloc(size, 64);

  float alpha = 1.0, beta = 0.0;

  vector<float> a(m * k);
  vector<float> b(k * n * groupSize);
  vector<float> c(k * n * groupSize);

  for (auto e : a) e = rand() / (float)(RAND_MAX / 9999);
  for (auto e : b) e = rand() / (float)(RAND_MAX / 9999);

  float *b_array[groupSize], *c_array[groupSize];

  for (int i = 0; i < groupSize; i++) {
    b_array[i] = b.data() + k * n * i;
    c_array[i] = c.data() + k * n * i;
  }
  double initial, end;
  initial = dsecnd();

  // tic();
  cblas_sgemm_pack(CblasRowMajor, CblasAMatrix, CblasNoTrans, m, n, k, alpha,
                   a.data(), k, Ap);

  for (int i = 0; i < groupSize; i++) {
    cblas_sgemm_compute(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k,
                        Ap, k, b_array[i], n, beta, c_array[i], n);
  }
  // cblas_sgemm_compute(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k,
  // a.data(), k, b.data(), n, beta, c.data(), n);
  // double elapsed = toc();
  end = dsecnd();

  double elapsed = end - initial;

  printf(
      " == Multiple Matrix multiplication (groupsize = %d, m n k = %d )using "
      "Intel(R) MKL cblas_sgemm "
      "completed == \n"
      " == at %.5f milliseconds == \n\n",
      Arg_G_Size, Arg_MKN, (elapsed * 1000));

  double sgemm_gflops = (2.0 * ((double)n) * ((double)m) * ((double)k) *
                         ((double)Arg_G_Size) * 1e-9);

  mkl_free(Ap);
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