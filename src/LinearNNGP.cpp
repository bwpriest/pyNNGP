#include "SeqNNGP.h"

#include "LinearNNGP.h"
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
LinearNNGP::LinearNNGP(const double* _y, const double* _X,
                       const double* _coords, const int _d, const int _q,
                       const int _p, const int _n, const int _m, CovModel& _cm,
                       DistFunc& _df, CompFunc& _cf, NoiseModel& _nm)
    : SeqNNGP(_y, _coords, _d, _q, _n, _m, _cm, _df, _cf, _nm),
      p(_p),
      Xt(_X, p, n) {
  // build the neighbor index
  nm.setX(Xt);

  // Klugy!
  assert(q == 1);
  beta = Xt.transpose()
             .bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV)
             .solve(y.row(0));
}

VectorXd LinearNNGP::additiveModel() const { return Xt.transpose() * beta; }

void LinearNNGP::updateW() {
  for (int i = 0; i < n; i++) {
    double a = 0.0;
    double v = 0.0;
    double e = 0.0;

    updateWparts(i, a, v, e);

    assert(q == 1);  // If q != 1 we should not be here. This is klugy and must
                     // be fixed.
    double mu =
        (y(0, i) - Xt.col(i).dot(beta)) * nm.invTauSq(i) + e / F_mat[i] + a;
    double var = 1.0 / (nm.invTauSq(i) + 1.0 / F_mat[i] + v);

    std::normal_distribution<> norm{mu * var, std::sqrt(var)};
    w_vec[i] = norm(gen);
  }
}

void LinearNNGP::sample(int nSamples) {
  for (int s = 0; s < nSamples; s++) {
    updateW();
    updateBeta();
    nm.update(*this);
    cm.update(*this);
  }
}

void LinearNNGP::updateBeta() {
  // Klugy!
  assert(q == 1);
  VectorXd tmp_p{nm.getXtW() * (y.row(0) - w_vec)};
  MatrixXd tmp_pp{nm.getXtWX()};

  // May be more efficient ways to do this...
  VectorXd mean = tmp_pp.llt().solve(tmp_p);
  MatrixXd cov  = tmp_pp.inverse();
  beta          = MVNorm(mean, cov)(gen);
}

}  // namespace pyNNGP
