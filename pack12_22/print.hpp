#include <iostream>
#include <vector>

using namespace std;

void printMatrixA(const float* matrix, int l, int m) {
  for (int i = 0; i < l; i++) {
    for (int j = 0; j < m; j++) {
      cout << matrix[i * m + j] << " ";
    }
    cout << endl;
  }
}

void printVector(vector<float> ve, int m) {
  for (int i = 0; i < ve.size(); i++) {
    cout << ve[i] << " ";
    if ((i + 1) % m == 0) cout << endl;
    if ((i + 1) % (m * m)==0) cout << endl;
  }
  cout << endl;
}