#ifndef NNGP_SeqNNGP_h
#define NNGP_SeqNNGP_h

#include <Eigen/Dense>
#include <random>
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Nearest-Neighbor Gaussian Process. Based upon [DBFG16].
 */
namespace pyNNGP {
class CovModel;
class NoiseModel;
class DistFunc;
class SeqNNGP {
 public:
  SeqNNGP(const double* _y, const double* _coords, const int _d, const int _n,
          const int _m, CovModel& _cm, DistFunc& _df, NoiseModel& _nm);

  // Allocate our own memory for these
  // Nearest neighbors index.  Holds the indices of the neighbors of each node.
  std::vector<int> nnIndx;  // [nIndx]

  // Nearest neighbors ranges
  // Lower part holds starting index for each node
  // Upper part holds number of elements for each node
  std::vector<int> nnIndxLU;  // [2*n]

  // Distances between neighbors
  std::vector<double> nnDist;  // [nIndx]

  // Reverse index.  Holds which nodes have n as a neighbor
  std::vector<int> uIndx;  // [nIndx]

  // Ranges for reverse index.
  std::vector<int> uIndxLU;  // [2*n]
  // Which neighbor is it?
  std::vector<int> uiIndx;  // [nIndx]
  // Lower part holds cumulative sum of squared neighbors of 1:(i-1)
  // Uppoer part holds # of neighbors squared
  std::vector<int> CIndx;  // [2*n]

  const int d;      // Number of input dimensions
  const int n;      // Number of input locations
  const int m;      // Number of nearest neighbors
  const int nIndx;  // Total number of neighbors (DAG edges)

  // Use existing memory here (allocated in python-layer)
  const Eigen::Map<const VectorXd> y;       // [n]
  const Eigen::Map<const MatrixXd> coords;  // [d, n]  ([n, d] in python)

  CovModel& cm;    // Model for GP covariances
  NoiseModel& nm;  // Model for additional measurement noise
  DistFunc& df;

  std::random_device rd;
  std::mt19937 gen;

  // These are mostly internal, but I'm too lazy for the moment to make them
  // private We allocate this memory ourselves.
  std::vector<double> B;      // [nIndx]
  std::vector<double> F;      // [n]
  std::vector<double> Bcand;  // [nIndx]
  std::vector<double> Fcand;  // [n]
  // stacked q x mq cross-covariance matrices C_{s_i, N(s_i)} between w(s_i) and
  // w_{N(s_i)}.
  // We appear to be assuming q = 1 for now.
  std::vector<double> c;  // [nIndx]
  // stacked ~ (m x m) covariance matrices C_{N(s_i)}.
  std::vector<double> C;  // [~n*m*m]
  // stacked ~ (m x m) cross pairwise distances for N(s_i).
  std::vector<double> D;  // [~n*m*m]
  VectorXd w;             // [n] Latent GP samples
  VectorXd beta;          // [p] Unknown linear model coefficients

  // return the additive model against which the GP is modeling discrepency.
  // This is the zero vector for the raw NNGP.
  virtual VectorXd additiveModel() const { return VectorXd::Zero(n); }

  virtual void sample(int nSamples);  // One Gibbs iteration

  // Use a particular covariance model to update given B and F vectors.
  void updateBF(double*, double*, CovModel&);
  virtual void updateW();

  void predict(const double* X0, const double* coords, const int* nnIndx0,
               int q, double* w0, double* y0);

 protected:
  void mkUIndx();
  void mkUIIndx();
  void mkCD();
  void updateWparts(const int i, double& a, double& v, double& e);
};

}  // namespace pyNNGP

#endif
