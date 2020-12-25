#include <sys/time.h>
#include <time.h>
#include <fstream>
#include <iostream>
#include <vector>
#include "mkl.h"
#include "print.hpp"

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

  float alpha = 1.0, beta = 0.0;

  vector<float> a(m * k, 1);
  vector<float> b(k * n * groupSize, 1);
  vector<float> c(k * n * groupSize, 0);

  // printMatrixA((float*)a.data(), m, k); printMatrixBC(b, k, n,
  // groupSize); printMatrixBC(c, m, n, groupSize);

  // printVector(a, k);
  // printVector(b, n);
  // printVector(c, n);

  cout << "------------------computing---------------" << endl;

  const float* b_array[groupSize];
  float* c_array[groupSize];

  for (int i = 0; i < groupSize; i++) {
    b_array[i] = b.data() + (i * k * n);
    c_array[i] = c.data() + (i * m * n);
  }

  // size_t : unsigned long
  size_t Asize = cblas_sgemm_pack_get_size(CblasAMatrix, m, n, k);
  // CBLA_IDENTIFIER:Specifies which matrix is to be packed:
  // If identifier = CblasAMatrix, the size returned is the size required to
  // store matrix A in an internal format.
  // If identifier = CblasBMatrix, the sizereturned is the size required to
  // store matrix B in an internal format.
  float* Ap = (float*)mkl_malloc(Asize, 32);

  // for (int i = 0; i < a.size(); i++) a[i] = rand() / (float)(RAND_MAX /
  // 9999); for (int i = 0; i < b.size(); i++) b[i] = rand() / (float)(RAND_MAX
  // / 9999);

  double initial1, end1;
  initial1 = dsecnd();

  for (int i = 0; i < groupSize; i++) {
    cblas_sgemm_compute(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k,
                        a.data(), k, b_array[i], n, beta, c_array[i], n);
  }
  end1 = dsecnd();
  double elapsed1 = end1 - initial1;
  printVector(c, n);
  cout << endl;

  //----------------------------------compare pack----------------------

  // for (int i = 0; i < a.size(); i++) a[i] = rand() / (float)(RAND_MAX / 9999); 
  // for (int i = 0; i < b.size(); i++) b[i] = rand() / (float)(RAND_MAX / 9999);

  double initial2, end2;
  initial2 = dsecnd();

  cblas_sgemm_pack(CblasRowMajor, CblasAMatrix, CblasNoTrans, m, n, k, alpha,
                   a.data(), k, Ap);

  for (int i = 0; i < groupSize; i++) {
    cblas_sgemm_compute(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k,
                        a.data(), k, b_array[i], n, beta, c_array[i], n);
  }
  end2 = dsecnd();
  double elapsed2 = end2 - initial2;

  printVector(c, n);
  cout << endl;

  printf(
      " == Multiple Matrix multiplication (groupsize = %d, m n k = %d )using "
      "Intel(R) MKL cblas_sgemm "
      "completed == \n"
      " == at %.5f milliseconds == \n\n",
      Arg_G_Size, Arg_MKN, (elapsed1 * 1000));
  cout << "--------------------------------after "
          "pack--------------------------------"
       << endl;
  printf(
      " == Multiple Matrix multiplication (groupsize = %d, m n k = %d )using"
      "Intel(R) MKL cblas_sgemm "
      " completed == \n"
      " == at %.5f milliseconds == \n\n",
      Arg_G_Size, Arg_MKN, (elapsed2 * 1000));

  double sgemm_gflops = (2.0 * ((double)n) * ((double)m) * ((double)k) *
                         ((double)Arg_G_Size) * 1e-9);

  mkl_free(Ap);

  cout << "Gflops" << sgemm_gflops / elapsed1 << "    ";
  cout << "Gflops" << sgemm_gflops / elapsed2 << "    ";
}

// 就是频率 向量长度 2 核数 乘起来
// 这个是计算sgemm的计算量,这个值除以理论峰值就是效率