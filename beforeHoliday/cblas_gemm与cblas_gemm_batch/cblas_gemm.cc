#include <stdio.h>

#include <fstream>
#include <iostream>
#include <vector>

#include "mkl.h"
using namespace std;

int main(int argc, char **argv) {
  // mkl_set_num_threads(1);

  MKL_INT Arg_G_Size = atoi(argv[1]);
  MKL_INT Arg_MKN_value = atoi(argv[2]);

#define groupCount 1

  MKL_INT groupSize = Arg_G_Size;

  MKL_INT m_value, k_value, n_value;

  m_value = k_value = n_value = Arg_MKN_value;

  srand(time(0));

  float *a =
      (float *)malloc((size_t)sizeof(float) * m_value * k_value * groupSize);
  float *b =
      (float *)malloc((size_t)sizeof(float) * k_value * n_value * groupSize);
  float *c =
      (float *)malloc((size_t)sizeof(float) * m_value * n_value * groupSize);

  for (int i = 0; i < m_value * k_value * Arg_G_Size; ++i) {
    a[i] = rand() / (float)(RAND_MAX / 9999);
    b[i] = rand() / (float)(RAND_MAX / 9999);
  }

  MKL_INT lda, ldb, ldc;
  lda = ldb = ldc = m_value;

  CBLAS_TRANSPOSE transA = CblasNoTrans;  // A
  CBLAS_TRANSPOSE transB = CblasNoTrans;  // B转置

  float alpha = 1.0;  // C=alpha*A*Btrans+C*beta
  float beta = 0.0;

  // const MKL_INT size_per_grp = 4;

  const float *a_array[groupSize], *b_array[groupSize];
  float *c_array[groupSize];
  for (int i = 0; i < groupSize;
       ++i) {                                //标记array[i]指向的数组开始的位置,现在只有一个group分为4个sub
    a_array[i] = a + i * m_value * k_value;  //指针操作
    b_array[i] = b + i * k_value * n_value;
    c_array[i] = c + i * m_value * n_value;
  }

  double s_initial, s_elapsed;  //时间

  s_initial = dsecnd();

  for (int i = 0; i < groupSize; i++) {
    cblas_sgemm(CblasRowMajor, transA, transB, m_value, n_value, k_value, alpha,
                a_array[i], lda, b_array[i], ldb, beta, c_array[i], ldc);
  }

  s_elapsed = dsecnd() - s_initial;

  double sgemm_gflops = (2.0 * ((double)n_value) * ((double)m_value) *
                         ((double)k_value) * ((double)Arg_G_Size) * 1e-9);

  ofstream writeTime, writeGflops, writeForGflops;
  writeTime.open("recordTime.log", ios::app);
  writeGflops.open("recordGflops.log", ios::app);
  writeForGflops.open("writeForGflops.log", ios::app);

  // write << s_elapsed * 1000 << "    ";
  writeGflops << sgemm_gflops / s_elapsed << "----";
  writeForGflops << sgemm_gflops / s_elapsed << "----";
  writeTime << s_elapsed << "    ";

  writeGflops.close();
  writeTime.close();
  writeForGflops.close();

  printf(
      " == Multiple Matrix multiplication (groupsize = %d, m n k = %d )using "
      "Intel(R) MKL cblas_sgemm "
      "completed == \n"
      " == at %.5f milliseconds == \n\n",
      Arg_G_Size, Arg_MKN_value, (s_elapsed * 1000));
  free(a);
  free(b);
  free(c);
}