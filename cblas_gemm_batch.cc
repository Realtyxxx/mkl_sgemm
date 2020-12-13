#include <stdio.h>

#include <fstream>
#include <iostream>
#include <vector>

#include "mkl.h"
using namespace std;

int main(int argc, char **argv) {
  MKL_INT Arg_G_Size = atoi(argv[1]);
  MKL_INT Arg_MKN_value = atoi(argv[2]);

#define groupCount 1

  MKL_INT groupSize[groupCount] = {Arg_G_Size};

  MKL_INT m_value, k_value, n_value;

  m_value = k_value = n_value = Arg_MKN_value;

  MKL_INT m_array[groupCount] = {m_value};
  MKL_INT n_array[groupCount] = {n_value};
  MKL_INT k_array[groupCount] = {k_value};

  std::vector<float> a(m_value * k_value * Arg_G_Size);
  std::vector<float> b(k_value * n_value * Arg_G_Size);
  std::vector<float> c(m_value * n_value * Arg_G_Size);

  for (int i = 0; i < m_value * k_value * Arg_G_Size; ++i) {
    a[i] = int(rand());
    b[i] = int(rand());
  }

  MKL_INT lda[groupCount], ldb[groupCount], ldc[groupCount];
  lda[0] = ldb[0] = ldc[0] = m_value;

  CBLAS_TRANSPOSE transA[groupCount] = {CblasNoTrans};  // A
  CBLAS_TRANSPOSE transB[groupCount] = {CblasNoTrans};  // B转置

  float alpha[groupCount] = {1.0};  // C=alpha*A*Btrans+C*beta
  float beta[groupCount] = {0.0};

  // const MKL_INT size_per_grp = 4;

  const float *a_array[Arg_G_Size], *b_array[Arg_G_Size];
  float *c_array[Arg_G_Size];
  for (int i = 0; i < Arg_G_Size;
       ++i) {  //标记array[i]指向的数组开始的位置,现在只有一个group分为4个sub
    a_array[i] = a.data() + i * m_value * k_value;  //指针操作
    b_array[i] = b.data() + i * k_value * n_value;
    c_array[i] = c.data() + i * m_value * n_value;
  }

  double s_initial, s_elapsed;  //时间

  s_initial = dsecnd();

  cblas_sgemm_batch(CblasRowMajor, transA, transB, m_array, n_array, k_array,
                    alpha, a_array, lda, b_array, ldb, beta, c_array, ldc,
                    groupCount, groupSize);

  s_elapsed = dsecnd() - s_initial;

  ofstream write;
  write.open("cblas_gemm_batch.txt", ios::app);

  write << s_elapsed * 1000 << "    ";
  write.close();

  printf(
      " == Multiple Matrix multiplication (groupsize = %d, m n k = %d )using "
      "Intel(R) MKL cblas_sgemm_batch "
      "completed == \n"
      " == at %.5f milliseconds == \n\n",
      Arg_G_Size, Arg_MKN_value, (s_elapsed * 1000));

  //输出矩阵A
  // cout << "Arg_G_Size : " << Arg_G_Size << endl;
  // cout << Arg_G_Size << " A(" << m_value << "," << k_value << ')' << endl;
  // for (int i = 0; i < a.size(); ++i) {
  //   cout << a[i] << ' ';
  //   if ((i + 1) % m_value == 0) cout << endl;
  //   if ((i + 1) % m_value * k_value == 0) {
  //     cout << endl;
  //     cout << endl;
  //   }
  // }

  // cout << Arg_G_Size << " B(" << k_value << "," << n_value << ')' << endl;
  // for (int i = 0; i < b.size(); ++i) {
  //   cout << b[i] << ' ';
  //   if ((i + 1) % k_value == 0) cout << endl;
  //   if ((i + 1) % k_value * n_value == 0) {
  //     cout << endl;
  //     cout << endl;
  //   }
  // }

  // cout << Arg_G_Size << " C(" << m_value << "," << n_value << ')' << endl;
  // for (int i = 0; i < c.size(); ++i) {
  //   cout << c[i] << ' ';
  //   if ((i + 1) % m_value == 0) cout << endl;
  //   if ((i + 1) % m_value * n_value == 0) {
  //     cout << endl;
  //     cout << endl;
  //   }
  // }

  // printf(
  //     " == Multiple Matrix multiplication using Intel(R) MKL cblas_sgemm with
  //     " "for loop completed == \n" " == at %.5f milliseconds == \n\n",
  //     (s_elapsed * 1000));
}

// std::cout << "a.size(): " << a.size() << std::endl;
// for (int i = 0; i < a.size(); ++i) {
//   std::cout << a[i] << " ";
//   if ((i + 1) % 40 == 0) std::cout << std::endl;
// }

// std::cout << "b.size(): " << b.size() << std::endl;
// for (int i = 0; i < b.size(); ++i) {
//   std::cout << b[i] << " ";
//   if ((i + 1) % 40 == 0) std::cout << std::endl;
// }

// std::cout << "c_array.size(): " << 20 * 20 * 4 << std::endl;
// for (int i = 0; i < 1600; ++i) {
//   std::cout << c[i] << " ";
//   if ((i + 1) % 80 == 0) std::cout << std::endl;
// }
// std::cout << std::endl;
// return 0;