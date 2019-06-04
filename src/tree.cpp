#include "tree.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>
#include "utils.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace pyNNGP {
void getNNIndx(int i, int m, int& iNNIndx, int& iNN) {
  // return index into nnIndx array, nnDist array, ...
  // iNN is # of neighbors of i
  if (i == 0) {
    iNNIndx = 0;  // this should never be accessed
    iNN = 0;
    return;
  } else if (i < m) {
    iNNIndx = (i * (i - 1)) / 2;
    iNN = i;
    return;
  } else {
    iNNIndx = (m * (m - 1)) / 2 + (i - m) * m;
    iNN = m;
    return;
  }
}

Node::Node(int i) : index(i), left(nullptr), right(nullptr) {}

Node* miniInsert(Node* Tree, const MatrixXd& coords, const int index,
                 const int dim, const int d) {
  // 2D-tree
  if (!Tree) return new Node(index);

  if (coords(dim, index) <= coords(dim, Tree->index)) {
    Tree->left = miniInsert(Tree->left, coords, index, (dim + 1) % d, d);
  } else {
    Tree->right = miniInsert(Tree->right, coords, index, (dim + 1) % d, d);
  }
  return Tree;
}

void get_nn(Node* Tree, const int index, const int dim, const int d,
            const DistFunc& df, const MatrixXd& coords, double* nnDist,
            int* nnIndx, int iNNIndx, int iNN) {
  // input: Tree, index, d, coords
  // output: nnDist, nnIndx
  if (!Tree) return;

  double disttemp = df(coords.col(index), coords.col(Tree->index));

  if (index != Tree->index && disttemp < nnDist[iNNIndx + iNN - 1]) {
    nnDist[iNNIndx + iNN - 1] = disttemp;
    nnIndx[iNNIndx + iNN - 1] = Tree->index;
    rsort_with_index(&nnDist[iNNIndx], &nnIndx[iNNIndx], iNN);
  }

  Node* temp1 = Tree->left;
  Node* temp2 = Tree->right;

  if (coords(dim, index) > coords(dim, Tree->index)) std::swap(temp1, temp2);
  get_nn(temp1, index, (dim + 1) % d, d, df, coords, nnDist, nnIndx, iNNIndx,
         iNN);
  if (fabs(coords(dim, Tree->index) - coords(dim, index)) >
      nnDist[iNNIndx + iNN - 1])
    return;
  get_nn(temp2, index, (dim + 1) % d, d, df, coords, nnDist, nnIndx, iNNIndx,
         iNN);
}

void mkNNIndxTree0(const int n, const int m, const int d, const DistFunc& df,
                   const MatrixXd& coords, int* nnIndx, double* nnDist,
                   int* nnIndxLU) {
  int i, iNNIndx, iNN;
  double distance;
  int nIndx = ((1 + m) * m) / 2 + (n - m - 1) * m;
  // Results seem to depend on BUCKETSIZE, which seems weird...
  int BUCKETSIZE = 10;

  std::fill(&nnDist[0], &nnDist[0] + nIndx,
            std::numeric_limits<double>::infinity());

  Node* Tree = nullptr;
  int time_through = -1;

  for (i = 0; i < n; i++) {
    getNNIndx(i, m, iNNIndx, iNN);
    nnIndxLU[i] = iNNIndx;
    nnIndxLU[n + i] = iNN;
    if (time_through == -1) { time_through = i; }

    if (i != 0) {
      for (int j = time_through; j < i; j++) {
        getNNIndx(i, m, iNNIndx, iNN);
        distance = df(coords.col(i), coords.col(j));
        if (distance < nnDist[iNNIndx + iNN - 1]) {
          nnDist[iNNIndx + iNN - 1] = distance;
          nnIndx[iNNIndx + iNN - 1] = j;
          rsort_with_index(&nnDist[iNNIndx], &nnIndx[iNNIndx], iNN);
        }
      }
      if (i % BUCKETSIZE == 0) {
#ifdef _OPENMP
#pragma omp parallel for private(iNNIndx, iNN)
#endif
        for (int j = time_through; j < time_through + BUCKETSIZE; j++) {
          getNNIndx(j, m, iNNIndx, iNN);
          get_nn(Tree, j, 0, d, df, coords, nnDist, nnIndx, iNNIndx, iNN);
        }

        for (int j = time_through; j < time_through + BUCKETSIZE; j++) {
          Tree = miniInsert(Tree, coords, j, 0, d);
        }

        time_through = -1;
      }
      if (i == n - 1) {
#ifdef _OPENMP
#pragma omp parallel for private(iNNIndx, iNN)
#endif
        for (int j = time_through; j < n; j++) {
          getNNIndx(j, m, iNNIndx, iNN);
          get_nn(Tree, j, 0, d, df, coords, nnDist, nnIndx, iNNIndx, iNN);
        }
      }
    } else {  // i==0
      Tree = miniInsert(Tree, coords, i, 0, d);
      time_through = -1;
    }
  }
  delete Tree;
}
}  // namespace pyNNGP
