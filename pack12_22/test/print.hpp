#include <iostream>
#include <vector>

using namespace std;


void printVector(float * ve, int m, int groupSize) {
  int len = m * m * groupSize;
  for (int i = 0; i < len; i++) {
    cout << ve[i] << " ";
    if ((i + 1) % m == 0) cout << endl;
    if ((i + 1) % (m * m)==0) cout << endl;
  }
  cout << endl;
}