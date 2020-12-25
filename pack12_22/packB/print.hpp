#include <iostream>
#include <vector>

using namespace std;


void printVector(vector<float> ve, int m) {
  for (int i = 0; i < ve.size(); i++) {
    cout << ve[i] << " ";
    if ((i + 1) % m == 0) cout << endl;
    if ((i + 1) % (m * m)==0) cout << endl;
  }
  cout << endl;
}