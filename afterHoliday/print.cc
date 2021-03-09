#include <iostream>

void printVector(float* ve, int m ,int size) {
  for (int i = 0; i < size; i++) {
    std::cout << ve[i] << " ";
    if ((i + 1) % m == 0) std::cout << std::endl;
  }
  std::cout << std::endl;
}