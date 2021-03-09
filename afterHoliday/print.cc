#include <iostream>
#include <vector>

void printVector(std::vector<float> ve, int m) {
  for (int i = 0; i < int(ve.size()); i++) {
    std::cout << ve[i] << " ";
    if ((i + 1) % m == 0) std::cout << std::endl;
  }
  std::cout << std::endl;
}