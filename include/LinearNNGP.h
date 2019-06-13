#ifndef NNGP_LinearNNGP_h
#define NNGP_LinearNNGP_h

#include <Eigen/Dense>
#include <random>
#include <vector>
#include "SeqNNGP.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace pyNNGP {
class CovModel;
class NoiseModel;
class DistFunc;

/**
 * Implements the hierarchical linear spatially-varying regression model Eq. (3)
 * in Datta16.
 */
class LinearNNGP : public SeqNNGP {
 public:
  LinearNNGP(const double* _y, const double* _X, const double* _coords, int _d,
             int _p, int _n, int _m, CovModel& _cm, DistFunc& _df,
             NoiseModel& _nm);

  const int p;  // Number of indicators per input location

  // fixed l x p spatially-referenced predictors. Seem to assuming l = 1.
  const Eigen::Map<const MatrixXd> Xt;  // [p, n]  ([n, p] in python)
  // Unknown linear coefficients
  VectorXd beta;  // [p]

  VectorXd additiveModel() const override;
  void     sample(int nSamples) override;
  void     updateW() override;

  void updateBeta();

  void predict(const double* X0, const double* coords, const int* nnIndx0,
               int q, double* w0, double* y0);
};

}  // namespace pyNNGP

#endif
