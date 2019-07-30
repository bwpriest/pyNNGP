#ifndef NNGP_covModel_h
#define NNGP_covModel_h

#define _USE_MATH_DEFINES

#include <Eigen/Dense>
#include <cmath>
#include <random>
#include "SeqNNGP.h"
#include "utils.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace pyNNGP {

class CovModel {
 public:
  CovModel() {}

  virtual double cov(double) const = 0;

  virtual void update(SeqNNGP& seq) = 0;

  virtual ~CovModel() {}
};

// *****************************************************************************
// NNGP Kernels
// *****************************************************************************

/**
 * The neural network gaussian process covariance function from
 *
 * [3] Deep Neural Networks as Gaussian Processes
 * (https://arxiv.org/abs/1711.00165)
 *
 * We currently assume that variance parameters are fixed, and
 * that the network uses reLU activations.
 */

class NeuralNetworkCovModel : public CovModel {
 public:
  NeuralNetworkCovModel(const int L, const double sigmaSqW,
                        const double sigmaSqB)
      : _L(L), _sigmaSqW(sigmaSqW), _sigmaSqB(sigmaSqB), Kxx(L) {
    Kxx[0] = get_K_0(1.0);  // Assuming all inputs normalized.
    for (int ell = 1; ell < _L; ++ell) {
      Kxx[ell] = get_K_l(Kxx[ell - 1], ell - 1);
    }
  }

  inline double              getSigmaSqB() const { return _sigmaSqB; }
  inline double              getSigmaSqW() const { return _sigmaSqW; }
  inline std::vector<double> getKxx() const { return Kxx; }

  /**
   * We assume that the hyperparameters are fixed for now.
   */
  void update(SeqNNGP& seq) override {
    seq.updateBF(&seq.B_mat[0], &seq.F_mat[0], *this);
  };

  /**
   * Calls the NNGP covariance function.
   *
   * NEEDS REFACTOR: must get both x and x^\prime.
   */
  double cov(double x) const override {
    // assert(x >= 0 && x <= 1);
    if (x > 1.0) { x = 1.0; }
    double Kuv = get_K_0(x);
    for (int ell = 1; ell < _L; ++ell) { Kuv = get_K_l(Kuv, ell - 1); }
    return Kuv;
  }

 protected:
  /**
   * Assuming that all inputs have norm 1.
   */
  constexpr double get_K_0(const double x) const {
    return _sigmaSqB + _sigmaSqW * x;
  }

  /**
   * First part of Eq. (11) from [3].
   */
  inline double get_K_l(const double Kuv, const int ell) const {
    static const double sig_mod = _sigmaSqW / (2 * M_PI);
    double              theta   = get_theta(Kuv, ell);
    // double theta = 0;
    return _sigmaSqB + sig_mod * Kxx[ell] *
                           (std::sin(theta) + (M_PI - theta) * std::cos(theta));
  }

  /**
   * Second part of Eq. (11) from [3].
   */
  inline double get_theta(const double Kuv, const int ell) const {
    return std::acos(Kuv / Kxx[ell]);
  }

  const int _L;
  // Not const because we might want to implement sampling later
  double _sigmaSqW;
  double _sigmaSqB;

  // Stores the self-kernel values at each recursive level. Only needed once as
  // inputs are assumed to be normalized.
  std::vector<double> Kxx;
};

// *****************************************************************************
// Isometric Kernels
// *****************************************************************************

class IsotropicCovModel : public CovModel {
 public:
  IsotropicCovModel(double sigmaSq, double phi, const double phiUnifa,
                    const double phiUnifb, const double phiTuning,
                    const double sigmaSqIGa, const double sigmaSqIGb)
      : _sigmaSq(sigmaSq),
        _phi(phi),
        _phiUnifa(phiUnifa),
        _phiUnifb(phiUnifb),
        _phiTuning(phiTuning),
        _sigmaSqIGa(sigmaSqIGa),
        _sigmaSqIGb(sigmaSqIGb) {}

  void   setSigmaSq(double sigmaSq) { _sigmaSq = sigmaSq; }
  double getSigmaSq() { return _sigmaSq; }

  virtual void setPhi(double phi) { _phi = phi; }
  double       getPhi() { return _phi; }

  virtual void updateSigmaSq(SeqNNGP& seq) {
    double a = 0.0;
    double e = 0.0;
    double b = 0.0;
    int    j = 0;
#ifdef _OPENMP
#pragma omp parallel for private(e, j, b) reduction(+ : a)
#endif
    for (int i = 0; i < seq.n; i++) {
      b = seq.w_vec[i];
      if (seq.nnIndxLU[seq.n + i] > 0) {
        e = 0.0;
        for (j = 0; j < seq.nnIndxLU[seq.n + i]; j++) {
          e += seq.B_mat[seq.nnIndxLU[i] + j] *
               seq.w_vec[seq.nnIndx[seq.nnIndxLU[i] + j]];
        }
        b -= e;
      }
      a += b * b / seq.F_mat[i];
    }

    std::gamma_distribution<> gamma{_sigmaSqIGa + seq.n / 2.0,
                                    _sigmaSqIGb + 0.5 * a * _sigmaSq};
    // _sigmaSq = 1.0/gamma(seq.gen);
  }

  virtual void updatePhi(SeqNNGP& seq) {
    double phiCurrent = getPhi();
    seq.updateBF(&seq.B_mat[0], &seq.F_mat[0], *this);
    double a      = 0.0;
    double b      = 0.0;
    double e      = 0.0;
    double logDet = 0.0;
    int    j      = 0;

// Get the current log determinant
#ifdef _OPENMP
#pragma omp parallel for private(e, j, b) reduction(+ : a, logDet)
#endif
    for (int i = 0; i < seq.n; i++) {
      b = seq.w_vec[i];
      if (seq.nnIndxLU[seq.n + i] > 0) {
        e = 0.0;
        for (j = 0; j < seq.nnIndxLU[seq.n + i]; j++) {
          e += seq.B_mat[seq.nnIndxLU[i] + j] *
               seq.w_vec[seq.nnIndx[seq.nnIndxLU[i] + j]];
        }
        b -= e;
      }
      a += b * b / seq.F_mat[i];
      logDet += std::log(seq.F_mat[i]);
    }
    double logPostCurrent = -0.5 * logDet - 0.5 * a;
    logPostCurrent += std::log(_phi - _phiUnifa) + std::log(_phiUnifb - _phi);

    // candidate
    std::normal_distribution<> norm{logit(_phi, _phiUnifa, _phiUnifb),
                                    _phiTuning};
    double phiCand = logitInv(norm(seq.gen), _phiUnifa, _phiUnifb);
    // Careful!!  Modifying *this.  Need to unmodify if proposal is not
    // accepted.
    setPhi(phiCand);

    seq.updateBF(&seq.Bcand[0], &seq.Fcand[0], *this);

    a      = 0.0;
    logDet = 0.0;

#ifdef _OPENMP
#pragma omp parallel for private(e, j, b) reduction(+ : a, logDet)
#endif
    for (int i = 0; i < seq.n; i++) {
      double b = seq.w_vec[i];
      if (seq.nnIndxLU[seq.n + i] > 0) {
        double e = 0.0;
        for (int j = 0; j < seq.nnIndxLU[seq.n + i]; j++) {
          e += seq.Bcand[seq.nnIndxLU[i] + j] *
               seq.w_vec[seq.nnIndx[seq.nnIndxLU[i] + j]];
        }
        b -= e;
      }
      a += b * b / seq.Fcand[i];
      logDet += std::log(seq.Fcand[i]);
    }

    double logPostCand = -0.5 * logDet - 0.5 * a;
    logPostCand +=
        std::log(phiCand - _phiUnifa) + std::log(_phiUnifb - phiCand);

    std::uniform_real_distribution<> unif{0.0, 1.0};
    if (unif(seq.gen) <= std::exp(logPostCand - logPostCurrent)) {
      std::swap(seq.B_mat, seq.Bcand);
      std::swap(seq.F_mat, seq.Fcand);
      // phiCand already set.
    } else {
      setPhi(phiCurrent);
    }
  }

  void update(SeqNNGP& seq) override {
    updateSigmaSq(seq);
    updatePhi(seq);
  }

 protected:
  double       _sigmaSq;
  double       _phi;
  const double _phiUnifa, _phiUnifb;      // Uniform prior on phi
  const double _phiTuning;                // Width of phi proposal distribution
  const double _sigmaSqIGa, _sigmaSqIGb;  // Inverse gamma prior on sigmaSq
};

class ExponentialCovModel : public IsotropicCovModel {
 public:
  using IsotropicCovModel::IsotropicCovModel;

  double cov(double x) const override { return _sigmaSq * std::exp(-x * _phi); }
};

class SphericalCovModel : public IsotropicCovModel {
 public:
  using IsotropicCovModel::IsotropicCovModel;

  double cov(double x) const override {
    if (x > 0.0 && x < _phiInv) {
      return _sigmaSq * (1.0 - 1.5 * _phi * x + 0.5 * std::pow(_phi * x, 3));
    } else if (x >= _phiInv) {
      return 0.0;
    } else {
      return _sigmaSq;
    }
  }

  void setPhi(double phi) override {
    _phi    = phi;
    _phiInv = 1. / phi;
  }

 private:
  double _phiInv;
};

class SqExpCovModel : public IsotropicCovModel {
 public:
  using IsotropicCovModel::IsotropicCovModel;

  double cov(double x) const override {
    return _sigmaSq * std::exp(-1.0 * std::pow(_phi * x, 2));
  }
};

}  // namespace pyNNGP

#endif
