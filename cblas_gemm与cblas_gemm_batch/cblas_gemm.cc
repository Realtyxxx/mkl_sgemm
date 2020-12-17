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

  MKL_INT groupSize = Arg_G_Size;

  MKL_INT m_value, k_value, n_value;

  m_value = k_value = n_value = Arg_MKN_value;

  std::vector<float> a(m_value * k_value * groupSize, 1.0);
  std::vector<float> b(k_value * n_value * groupSize, 1.0);
  std::vector<float> c(m_value * n_value * groupSize, 0.0);

  srand(time(0));
  for (int i = 0; i < m_value * k_value * Arg_G_Size; ++i) {
    a[i] = int(rand());
    b[i] = int(rand());
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
       ++i) {  //标记array[i]指向的数组开始的位置,现在只有一个group分为4个sub
    a_array[i] = a.data() + i * m_value * k_value;  //指针操作
    b_array[i] = b.data() + i * k_value * n_value;
    c_array[i] = c.data() + i * m_value * n_value;
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

  ofstream write;
  write.open("record.txt", ios::app);

  // write << s_elapsed * 1000 << "    ";
  write << sgemm_gflops / s_elapsed << "    ";
  write.close();

  printf(
      " == Multiple Matrix multiplication (groupsize = %d, m n k = %d )using "
      "Intel(R) MKL cblas_sgemm "
      "completed == \n"
      " == at %.5f milliseconds == \n\n",
      Arg_G_Size, Arg_MKN_value, (s_elapsed * 1000));

  // // 输出矩阵A
  // cout << "groupSize : " << groupSize << endl;
  // cout << groupSize << " A(" << m_value << "," << k_value << ')' << endl;
  // for (int i = 0; i < a.size(); ++i) {
  //   cout << a[i] << ' ';
  //   if ((i + 1) % m_value == 0) cout << endl;
  //   if ((i + 1) % m_value * k_value == 0) {
  //     cout << endl;
  //     cout << endl;
  //   }
  // }

  // cout << groupSize << " B(" << k_value << "," << n_value << ')' << endl;
  // for (int i = 0; i < b.size(); ++i) {
  //   cout << b[i] << ' ';
  //   if ((i + 1) % k_value == 0) cout << endl;
  //   if ((i + 1) % k_value * n_value == 0) {
  //     cout << endl;
  //     cout << endl;
  //   }
  // }

  // cout << groupSize << " C(" << m_value << "," << n_value << ')' << endl;
  // for (int i = 0; i < c.size(); ++i) {
  //   cout << c[i] << ' ';
  //   if ((i + 1) % m_value == 0) cout << endl;
  //   if ((i + 1) % m_value * n_value == 0) {
  //     cout << endl;
  //     cout << endl;
  //   }
  // }

  // printf(
  //     " == Multiple Matrix multiplication using Intel(R) MKL cblas_sgemm
  //     with" "for loop completed == \n" " == at %.5f milliseconds == \n\n",
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