#include <stdio.h>

#include <fstream>
#include <iostream>
#include <vector>

#include "mkl.h"

// mkl_set_num_threads(1);
// MKL_Set_Num_Threads(1);

void printVector(float *, int, int);

int main(int argc, char **argv) {
  // mkl_set_num_threads(1);

  MKL_INT Arg_G_Size = atoi(argv[1]);
  MKL_INT Arg_M_value = atoi(argv[2]);
  MKL_INT Arg_K_value = atoi(argv[3]);
  MKL_INT Arg_N_value = atoi(argv[4]);

#define groupCount 1
  //设置一种批次,批次数目为Arg_G_Size
  MKL_INT groupSize[groupCount] = {Arg_G_Size};

  MKL_INT m_value, k_value, n_value;
  //给m, n, k赋值
  m_value = Arg_M_value;
  k_value = Arg_K_value;
  n_value = Arg_N_value;

  MKL_INT m_array[groupCount] = {m_value};
  MKL_INT n_array[groupCount] = {n_value};
  MKL_INT k_array[groupCount] = {k_value};

  //分配内存数组形式储存值
  float *a =
      (float *)malloc((size_t)sizeof(float) * m_value * k_value * Arg_G_Size);
  float *b =
      (float *)malloc((size_t)sizeof(float) * k_value * n_value * Arg_G_Size);
  float *c =
      (float *)malloc((size_t)sizeof(float) * m_value * n_value * Arg_G_Size);

  srand(time(0));

  //给A,B矩阵赋值
  // Initialize the A matrix with random float number
  for (int i = 0; i < m_value * k_value * Arg_G_Size; ++i) {
    a[i] = rand() / (float)(RAND_MAX / 9999);
  }

  // Initialize the B matrix with random float number
  for (int i = 0; i < k_value * n_value * Arg_G_Size; ++i) {
    b[i] = rand() / (float)(RAND_MAX / 9999);
  }

  //赋值leader dimmension参数
  MKL_INT lda[groupCount], ldb[groupCount], ldc[groupCount];
  lda[0] = k_value;
  ldb[0] = n_value;
  ldc[0] = n_value;

  CBLAS_TRANSPOSE transA[groupCount] = {CblasNoTrans};  // A
  CBLAS_TRANSPOSE transB[groupCount] = {CblasNoTrans};  // B转置

  float alpha[groupCount] = {1.0};  // C=alpha*A*Btrans+C*beta
  float beta[groupCount] = {0.0};

  //由于batch中存储矩阵的参数是需要用二维数组,所以将前面的连续一维数组隔断开,用不同位置的地址分配指针.
  const float *a_array[Arg_G_Size], *b_array[Arg_G_Size];
  float *c_array[Arg_G_Size];

  for (int i = 0; i < Arg_G_Size;
       ++i) {                                //标记array[i]指向的数组开始的位置,现在只有一个group分为4个sub
    a_array[i] = a + i * m_value * k_value;  //指针操作
    b_array[i] = b + i * k_value * n_value;
    c_array[i] = c + i * m_value * n_value;
  }

  double s_initial, s_elapsed;  //时间

  // batch函数执行位置与掐时间位置
  s_initial = dsecnd();

  cblas_sgemm_batch(CblasRowMajor, transA, transB, m_array, n_array, k_array,
                    alpha, a_array, lda, b_array, ldb, beta, c_array, ldc,
                    groupCount, groupSize);

  s_elapsed = dsecnd() - s_initial;

  //打印测试
/*   printVector((float *)a_array[0], (int)k_value, (int)(m_value * k_value));
  printVector((float *)b_array[0], (int)n_value, (int)(k_value * n_value));
  printVector((float *)c_array[0], (int)n_value, (int)(m_value * n_value)); */

  double sgemm_gflops = (2.0 * ((double)n_value) * ((double)m_value) *
                         ((double)k_value) * ((double)Arg_G_Size) * 1e-9);

  std::ofstream writeTime, writeGflops, writeBatchGflops;
  writeTime.open("recordTime.log", std::ios::app);
  writeGflops.open("recordGflops.log", std::ios::app);
  writeBatchGflops.open("writeBatchGflops.log", std::ios::app);

  // write << s_elapsed * 1000 << "    ";
  writeGflops << sgemm_gflops / s_elapsed << "----";
  writeBatchGflops << sgemm_gflops / s_elapsed << "----";
  writeTime << s_elapsed << "    ";

  writeGflops.close();
  writeTime.close();
  writeBatchGflops.close();

  printf(
      " == Multiple Matrix multiplication (groupsize = %d, m = %d, n = %d, k = "
      "%d )using "
      "Intel(R) MKL cblas_sgemm_batch "
      "completed == \n"
      " == at %.5f milliseconds == \n\n",
      Arg_G_Size, Arg_M_value, Arg_N_value, Arg_K_value, (s_elapsed * 1000));
  free(a);
  free(b);
  free(c);
}