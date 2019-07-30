#include "SeqNNGP.h"
#include "covModel.h"
#include "distFunc.h"
#include "noiseModel.h"
#include "tree.h"
#include "utils.h"

#include <chrono>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;

namespace pyNNGP {
SeqNNGP::SeqNNGP(const double* _y_targets, const double* _coords,
                 const int _d_idim, const int _q_ldim, const int _n_samples,
                 const int _m_nns, CovModel& _cm, DistFunc& _df,
                 NoiseModel& _nm)
    : d(_d_idim),
      q(_q_ldim),
      n(_n_samples),
      m(_m_nns),
      nIndx(m * (m + 1) / 2 + (n - m - 1) * m),
      y(_y_targets, q, n),    // n x q in python is q x n in Eigen
      coords(_coords, d, n),  // n x d in python is d x n in Eigen
      cm(_cm),
      df(_df),
      nm(_nm),
      gen(rd()),
      regression_ready(false),
      regression_coeffs(Eigen::MatrixXd::Zero(q, n)),
      w_vec(VectorXd::Zero(n)) {
  // build the neighbor index
  nnIndx.resize(nIndx);
  nnDist.resize(nIndx);
  nnIndxLU.resize(2 * n);
  w_vec.resize(n);

  std::cout << "Finding neighbors" << '\n';
  auto start = std::chrono::high_resolution_clock::now();
  mkNNIndxTree0(n, m, d, df, coords, &nnIndx[0], &nnDist[0], &nnIndxLU[0]);
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> diff = end - start;
  std::cout << "duration = " << diff.count() << "s" << '\n';

  std::cout << "Building neighbors of neighbors index" << '\n';
  start = std::chrono::high_resolution_clock::now();
  mkUIndx();
  end  = std::chrono::high_resolution_clock::now();
  diff = end - start;
  std::cout << "duration = " << diff.count() << "s" << '\n';

  B_mat.resize(nIndx);
  Bcand.resize(nIndx);
  F_mat.resize(n);
  Fcand.resize(n);
  std::cout << "Making CD" << '\n';
  start = std::chrono::high_resolution_clock::now();
  mkCD();
  end  = std::chrono::high_resolution_clock::now();
  diff = end - start;
  std::cout << "duration = " << diff.count() << "s" << '\n';

  std::cout << "updating BF" << '\n';
  start = std::chrono::high_resolution_clock::now();
  updateBF(&B_mat[0], &F_mat[0], cm);
  end  = std::chrono::high_resolution_clock::now();
  diff = end - start;
  std::cout << "duration = " << diff.count() << "s" << '\n';

  std::cout << "initiating regression coefficients"
            << "\n";
  start = std::chrono::high_resolution_clock::now();
  regression_init();
  end  = std::chrono::high_resolution_clock::now();
  diff = end - start;
  std::cout << "duration = " << diff.count() << "s" << '\n';
}

void SeqNNGP::sample(int nSamples) {
  for (int s = 0; s < nSamples; s++) {
    updateW();
    nm.update(*this);
    cm.update(*this);
  }
}

void SeqNNGP::mkUIndx() {
  uIndx.reserve(nIndx);
  uiIndx.reserve(nIndx);
  uIndxLU.resize(2 * n);

  // Look through each coordinate (node)
  for (int i = 0; i < n; i++) {
    int k = 0;
    // Look through nodes that might have i as a neighbor (child)
    for (int j = i + 1; j < n; j++) {
      // Get start and end range of where to check nnIndx if i is a child of j
      int nnStart, nnEnd;
      if (j < m) {
        nnStart = j * (j - 1) / 2;
        nnEnd   = (j + 1) * j / 2;
      } else {
        nnStart = m * (m - 1) / 2 + m * (j - m);
        nnEnd   = nnStart + m;
      }
      // Actually do the search for i
      auto result = std::find(&nnIndx[nnStart], &nnIndx[nnEnd], i);
      if (result != &nnIndx[nnEnd]) {  // If found
        uIndx.push_back(j);            // Record that j is a parent of i
        uiIndx.push_back(int(
            result - &nnIndx[nnStart]));  // Record which of i's parent it is
        k++;  // Increment the number of nodes that have j as a parent
      }
    }
    uIndxLU[n + i] = k;  // Set the number of nodes that have j as a parent
  }
  uIndxLU[0] = 0;
  for (int i = 0; i < n - 1; i++) uIndxLU[i + 1] = uIndxLU[i] + uIndxLU[n + i];
}

void SeqNNGP::mkCD() {
  CIndx.resize(2 * n);
  int j = 0;
  for (int i = 0; i < n; i++) {  // zero should never be accessed
    j += nnIndxLU[n + i] * nnIndxLU[n + i];
    if (i == 0) {
      CIndx[n + i] = 0;
      CIndx[i]     = 0;
    } else {
      CIndx[n + i] =
          nnIndxLU[n + i] * nnIndxLU[n + i];       // # of neighbors squared
      CIndx[i] = CIndx[n + i - 1] + CIndx[i - 1];  // cumulative sum of above...
    }
  }
  // j is now the sum of squares of neighbor counts
  C_cov.resize(j);
  c_crosscov.resize(nIndx);
  D_dist.resize(j);

  // This piece is currently used only by an Isometric kernel. For other
  // kernels, this work is wasted.
  for (int i = 0; i < n; i++) {                  // for all elements
    for (int k = 0; k < nnIndxLU[n + i]; k++) {  // for all neighbors of i
      for (int ell = 0; ell <= k; ell++) {       // lower triangular elements
        int i1 = nnIndx[nnIndxLU[i] + k];
        int i2 = nnIndx[nnIndxLU[i] + ell];
        // Currently assuming that the kernel function depends upon the distance
        // measure, e.g. isotropic kernel functions are paired with euclidean
        // distance, whereas dot product kernel functions are paired implicitly
        // with cosine similarity. Failing to correctly pair kernel functions
        // with distance metrics will result in unexpected behavior.
        //
        // It might be best to hide this dependence by placing the DistFunc
        // inside the CovModel later.
        D_dist[CIndx[i] + ell * nnIndxLU[n + i] + k] =
            df(coords.col(i1), coords.col(i2));
      }
    }
  }
}

// Modified to ignore Matern covariance until I can figure out a way to get
// a cyl_bessel_k compiled.
void SeqNNGP::updateBF(double* B, double* F, CovModel& cm) {
  int                  k, ell;
  Eigen::Map<VectorXd> eigenB(&B[0], nIndx);
  Eigen::Map<VectorXd> eigenF(&F[0], n);

#ifdef _OPENMP
#pragma omp parallel for private(k, ell)
#endif
  for (int i = 0; i < n; i++) {
    if (i > 0) {
      // Construct C_cov and c_crosscov matrices that we'll Chosolve below
      // I think these are essentially the constituents of eq (3) of Datta++14
      // I.e., we're updating auto- and cross-covariances
      for (k = 0; k < nnIndxLU[n + i]; k++) {
        // int i1 = nnIndx[nnIndxLU[i] + k];
        // get cross-covariance between i and its kth neighbor
        c_crosscov[nnIndxLU[i] + k] = cm.cov(nnDist[nnIndxLU[i] + k]);
        assert(nnDist[nnIndxLU[i] + k] ==
               df(coords.col(i), coords.col(nnIndx[nnIndxLU[i] + k])));
        for (ell = 0; ell <= k; ell++) {
          // int i2 = nnIndx[nnIndxLU[i] + ell];
          // Get covariance between i's kth and (ell*m + k)th neighbor.
          C_cov[CIndx[i] + ell * nnIndxLU[n + i] + k] =
              cm.cov(D_dist[CIndx[i] + ell * nnIndxLU[n + i] + k]);
        }
      }
      // Note symmetric, so shouldn't matter if I screw up row/col major here.
      const Eigen::Map<const MatrixXd> eigenC_cov(
          &C_cov[CIndx[i]], nnIndxLU[i + n], nnIndxLU[i + n]);
      const Eigen::Map<const VectorXd> eigenc_crosscov(&c_crosscov[nnIndxLU[i]],
                                                       nnIndxLU[i + n]);
      // Might be good to figure out how to use solveInPlace here.
      auto Blocal = eigenB.segment(nnIndxLU[i], nnIndxLU[n + i]);
      Blocal      = eigenC_cov.llt().solve(eigenc_crosscov);
      eigenF[i]   = cm.cov(0.0) - Blocal.dot(eigenc_crosscov);
    } else {
      B[i] = 0;
      F[i] = cm.cov(0.0);
    }
  }
}

void SeqNNGP::updateWparts(const int i, double& a, double& v, double& e) {
  if (uIndxLU[n + i] > 0) {  // is i a neighbor for anybody
    for (int j = 0; j < uIndxLU[n + i]; j++) {
      // for each location neighboring i
      double b  = 0.0;
      int    jj = uIndx[uIndxLU[i] + j];            // index of i's jth neighbor
      for (int k = 0; k < nnIndxLU[n + jj]; k++) {  // for each neighboring jj
        int kk = nnIndx[nnIndxLU[jj] + k];  // index of jj's kth neighbor
        if (kk != i) {                      // if the neighbor of jj is not i
          // covariance between jj and kk and the random effect of kk
          b += B_mat[nnIndxLU[jj] + k] * w_vec[kk];
        }
      }
      a += B_mat[nnIndxLU[jj] + uiIndx[uIndxLU[i] + j]] * (w_vec[jj] - b) /
           F_mat[jj];
      v += pow(B_mat[nnIndxLU[jj] + uiIndx[uIndxLU[i] + j]], 2) / F_mat[jj];
    }
  }

  for (int j = 0; j < nnIndxLU[n + i]; j++) {
    e += B_mat[nnIndxLU[i] + j] * w_vec[nnIndx[nnIndxLU[i] + j]];
  }
}

void SeqNNGP::updateW() {
  for (int i = 0; i < n; i++) {
    double a = 0.0;
    double v = 0.0;
    double e = 0.0;
    updateWparts(i, a, v, e);

    assert(q == 1);  // If q != 1 we should not be here. This is klugy and must
                     // be fixed.
    double mu  = y(0, i) * nm.invTauSq(i) + e / F_mat[i] + a;
    double var = 1.0 / (nm.invTauSq(i) + 1.0 / F_mat[i] + v);

    std::normal_distribution<> norm{mu * var, std::sqrt(var)};
    w_vec[i] = norm(gen);
  }
}

double SeqNNGP::quadratic_form(const std::vector<double>& u,
                               const std::vector<double>& v) const {
  assert(u.size() == n);
  assert(v.size() == n);

  std::vector<double> results(n);
  results[0] = u[0] * v[0] / F_mat[0];
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 1; i < n; ++i) {
    double left  = u[i];
    double right = v[i];
    for (int j = 0; j < nnIndxLU[n + 1]; ++j) {  // for i's jth neighbor
      const int ij = nnIndxLU[i] + j;  // sparse address of i's jth neighbor
      int       jj = nnIndx[nnIndxLU[i] + j];  // index of i's jth neighbor
      left -= u[jj] * B_mat[ij];
      right -= v[jj] * B_mat[ij];
    }
    results[i] = left * right / F_mat[i];
  }
  return std::accumulate(std::begin(results), std::end(results), 0.0);
}

Eigen::MatrixXd SeqNNGP::get_regression_coeffs() { return regression_coeffs; }

Eigen::MatrixXd SeqNNGP::MAPPredict(
    const Eigen::Ref<const Eigen::MatrixXd>& Xstar) {
  const int nstar = Xstar.cols();
  const int dstar = Xstar.rows();
  assert(dstar == d);
  // if (!regression_ready) {
  //   regression_init();
  //   regression_ready = true;
  // }

  // [q, nstar] in python
  Eigen::MatrixXd eigenYstar = Eigen::MatrixXd::Zero(q, nstar);

  for (int i = 0; i < nstar; i++) {
    // TODO: currently computing dense cross covariance matrix. Should
    // sparsify to save effort.
    const Eigen::VectorXd crosscov = dense_crosscov(Xstar.col(i));
    eigenYstar.col(i)              = regression_univariate(crosscov);
  }
  // return Xstar;
  // Eigen::VectorXd v(2);
  // v(0) = d, v(1) = dstar;
  // return v;
  // return crosscov;
  return eigenYstar;
  // return regression_coeffs;
}

Eigen::VectorXd SeqNNGP::dense_crosscov(
    const Eigen::Ref<const Eigen::VectorXd>& newcoord) {
  // assert(newcoord.size() == d);
  Eigen::VectorXd crosscov = Eigen::VectorXd::Zero(n);
  // #ifdef _OPENMP
  // #pragma omp parallel for
  // #endif
  for (int i = 0; i < n; ++i) {
    crosscov(i) = cm.cov(df(newcoord, coords.col(i)));
  }
  return crosscov;
}

Eigen::VectorXd SeqNNGP::regression_univariate(const Eigen::VectorXd& u) const {
  assert(u.size() == n);

  Eigen::VectorXd results = u(0) * regression_coeffs.col(0) / F_mat[0];
  // #ifdef _OPENMP
  // #pragma omp parallel for
  // #endif
  for (int i = 1; i < n; ++i) {
    double left = u(i);
    for (int j = 0; j < nnIndxLU[n + 1]; ++j) {  // for i's jth neighbor
      // const int jj = uIndx[uIndxLU[i] + j];      // index of i's jth neighbor
      // left -= u(jj) * B_mat[nnIndxLU[jj]];
      const int ij = nnIndxLU[i] + j;  // sparse address of i's jth neighbor
      const int jj = nnIndx[ij];       // index of i's jth neighbor
      left -= u(jj) * B_mat[ij];
    }
    results += left * regression_coeffs.col(i) / F_mat[i];
  }
  return results;
}

void SeqNNGP::regression_init() {
  regression_coeffs = y;
  for (int i = 1; i < n; ++i) {
    // #ifdef _OPENMP
    // #pragma omp parallel for
    // #endif
    for (int j = 0; j < nnIndxLU[n + i]; ++j) {
      // const int jj = uIndx[uIndxLU[i] + j];  // index of i's jth neighbor
      // regression_coeffs.col(i) -= y.col(jj) * B_mat[nnIndxLU[jj]];
      const int ij = nnIndxLU[i] + j;  // sparse address of i's jth neighbor
      const int jj = nnIndx[ij];       // index of i's jth neighbor
      regression_coeffs.col(i) -= y.col(jj) * B_mat[ij];
      // for (int k = 0; k < q; ++k) {
      //   regression_coeffs(k, i) -= y(k, jj) * B_mat[jj];
      // }
    }
  }
  regression_ready = true;
}

// void SeqNNGP::updateTauSq() {
//     VectorXd tmp_n = y - w - Xt.transpose()*beta;
//     std::gamma_distribution<> gamma{tauSqIGa+n/2.0,
//     tauSqIGb+0.5*tmp_n.squaredNorm()}; tauSq = 1.0/gamma(gen);
// }
//
// // Ought to work to get a single sample of w/y for new points.  Note, this
// version doesn't
// // Parallelize over samples.  Could probably parallelize over points though?
// (But don't
// // get to reuse distance measurements efficiently that way...)
// void SeqNNGP::predict(const double* _X0, const double* _coords0, const int*
// _nnIndx0, int q,
//              double* w0, double* y0)
// {
//     const Eigen::Map<const MatrixXd> coords0(_coords0, 2, q);
//     const Eigen::Map<const MatrixXd> Xt0(_X0, p, q);
//     // Could probably make the following a MatrixXi since all points have
//     exactly m neighbors const Eigen::Map<const VectorXi> nnIndx0(_nnIndx0,
//     m*q);
//
//     MatrixXd C(m, m);
//     VectorXd c(m);
//     for(int i=0; i<q; i++) {
//         for(int k=0; k<m; k++) {
//             // double d = dist2(coords.col(nnIndx0[k+q*i]), coords0.col(i));
//             //???? and below? double d = dist2(coords.col(nnIndx0[i+q*k]),
//             coords0.col(i)); c[k] = cm.cov(d); for(int ell=0; ell<m; ell++) {
//                 d = dist2(coords.col(nnIndx0[i+q*k]), coords.col(i+q*ell));
//                 C(ell,k) = cm.cov(d);
//             }
//         }
//         auto tmp = C.llt().solve(c);
//         double d = 0.0;
//         for(int k=0; k<m; k++) {
//             d += tmp[k]*w[nnIndx0[i+q*k]];
//         }
//
//         w0[i] = std::normal_distribution<>{d, std::sqrt(cm.cov(0.0) -
//         tmp.dot(c))}(gen); y0[i] =
//         std::normal_distribution<>{Xt0.col(i).dot(beta)+w0[i],
//         std::sqrt(tauSq)}(gen);
//     }
// }

/**
 * Params:
 *
 * _coords0 : input query data points
 */
// void SeqNNGP::predict(const double* _coords0, const int* _nnIndx0, int q,
//                       double* w0, double* y0) {
//   // I think that 2 in the original should be d.
//   const Eigen::Map<const MatrixXd> coords0(_coords0, d, q);
//   const Eigen::Map<const VectorXi> nnIndx0(_nnIndx0, m * q);

//   MatrixXd C0(m, m);
//   VectorXd c0(m);
//   for (int i = 0; i < q; i++) {
//     for (int k = 0; k < m; k++) {
//       double dist = df(coords.col(nnIndx0[i + q * k]), coords0.col(i));
//       c0[k]       = cm.cov(dist);
//       for (int ell = 0; ell < m; ell++) {
//         dist = df(coords.col(nnIndx0[i + q * k]), coords.col(i + q * ell));
//         C0(ell, k) = cm.cov(d);
//       }
//     }
//     auto   tmp  = C0.llt().solve(c0);
//     double dist = 0.0;
//     for (int k = 0; k < m; k++) { dist += tmp[k] * w[nnIndx0[i + q * k]]; }
//   }
// }

}  // namespace pyNNGP

// Predict
// input:
//   - orig coords
//   - X0, coords0, nnIndx0 - predictors, locations, and nearest neighbors of
//   prediction points
//   - beta/theta/w samples
// Maybe do this in parallel to original sampling?
// What takes more RAM, the samples, or the input locations?
// Maybe don't store X0, coords0, nnIndx0 in the class itself??
