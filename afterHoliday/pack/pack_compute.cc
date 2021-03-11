// #include <sys/time.h>
// #include <time.h>
#include <fstream>
#include <iostream>
#include <vector>
#include "mkl.h"

void printVector(float *, int, int);

using namespace std;

int main(int argc, char **argv) {
  //从命令行读取参数
  MKL_INT Arg_G_Size = atoi(argv[1]);
  MKL_INT Arg_M_value = atoi(argv[2]);
  MKL_INT Arg_K_value = atoi(argv[3]);
  MKL_INT Arg_N_value = atoi(argv[4]);

  MKL_INT groupSize = Arg_G_Size;

  int m, k, n;
  m = Arg_M_value;
  k = Arg_K_value;
  n = Arg_N_value;

  float alpha = 1.0, beta = 0.0;

  float *a = (float *)malloc((size_t)sizeof(float) * m * k * groupSize);
  float *b = (float *)malloc((size_t)sizeof(float) * k * n);
  float *c = (float *)malloc((size_t)sizeof(float) * m * n * groupSize);

  // size_t : unsigned long
  size_t size = cblas_sgemm_pack_get_size(CblasBMatrix, m, n, k);
  float *Bp = (float *)mkl_malloc(size, 64);
  // CBLA_IDENTIFIER:Specifies which matrix is to be packed:
  // If identifier = CblasAMatrix, the size returned is the size required to
  // store matrix A in an internal format.
  // If identifier = CblasBMatrix, the sizereturned is the size required to
  // store matrix B in an internal format.

  int aSize = m * k * groupSize;
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
    a_array[i] = a + m * k * i;
    c_array[i] = c + m * n * i;
  }




  //掐时间了
  double initial, end;
  initial = dsecnd();

  cblas_sgemm_pack(CblasRowMajor, CblasBMatrix, CblasNoTrans, m, n, k, alpha, b,
                   n, Bp);

  for (int i = 0; i < groupSize; i++) {
    cblas_sgemm_compute(CblasRowMajor, CblasNoTrans, CblasPacked, m, n, k,
                        a_array[i], k, Bp, n, beta, c_array[i], n);
  }
  end = dsecnd();

  double elapsed = end - initial;



  //打印测试
  // printVector((float *)a_array[0], (int)k, (int)(m * k));
  // printVector((float *)b, (int)n, (int)(k * n));
  // printVector((float *)c_array[0], (int)n, (int)(m * n));



  double sgemm_gflops = (2.0 * ((double)n) * ((double)m) * ((double)k) *
                         ((double)Arg_G_Size) * 1e-9);

  printf(
      " == Multiple Matrix multiplication (groupsize = %d, m = %d, n = %d, k = "
      "%d )using "
      "Intel(R) MKL cblas_sgemm "
      "completed == \n"
      " == at %.5f milliseconds == \n",
      Arg_G_Size, m, n, k, (elapsed * 1000));

  // float total = 48;
  // float efficiency = sgemm_gflops / elapsed / total * 100 ;
  std::cout << "gflops : " << sgemm_gflops / elapsed << std::endl;
  // std::cout << "efficiency : " << efficiency << "% " << std::endl;

  free(a);
  free(b);
  free(c);

  mkl_free(Bp);

  ofstream writeGflops, writeRuntime, writeEfficiency;
  writeGflops.open("pack_compute_Gflops.log", ios::app);
  writeRuntime.open("pack_compute_Runtime.log", ios::app);
  // writeEfficiency.open("pack_compute_efficiency.log", ios::app);

  writeGflops << sgemm_gflops / elapsed << "-------";
  writeRuntime << elapsed * 1000 << "-------";
  // writeEfficiency << efficiency << "%    ";

  writeRuntime.close();
  writeGflops.close();
}

// 就是频率 向量长度 2 核数 乘起来
// 这个是计算sgemm的计算量,这个值除以理论峰值就是效率