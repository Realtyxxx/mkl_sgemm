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

void printMatrixBC(vector<float> matrix, int l, int m, int gsize) {
  float* m_array[gsize];
  for (int i = 0; i < gsize; i++) {
    m_array[i] = matrix.data() + l * m * i;
    cout << "number:" << i << endl;
    printMatrixA(m_array[i], l, m);
  }
}

void printVector(vector<float> ve , int m) {
  for(int i=0;i<ve.size();i++){
		cout<<ve[i]<<" ";
		if((i + 1)%m == 0) cout<<endl;
	}
	cout<<endl;
}