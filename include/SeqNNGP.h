#ifndef NNGP_SeqNNGP_h
#define NNGP_SeqNNGP_h

#include "FixedPriorityQueue.h"

#include <Eigen/Dense>
#include <map>
#include <random>
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Nearest-Neighbor Gaussian Process. Implementations based upon
 *
 * [1] Hierarchical Nearest-Neighbor Gaussian Process Models for Large
 * Geostatistical Datasets (https://arxiv.org/abs/1406.7343)
 *
 * [2] Efficient Algorithms for Bayesian Nearest Neighbor Gaussian Processes
 * (https://arxiv.org/abs/1702.00434)
 */
namespace pyNNGP {

typedef pyNNGP::operand<int>            nbr_t;
typedef pyNNGP::FixedPriorityQueue<int> fpq_t;
typedef std::map<int, double>           nbr_map_t;
typedef std::vector<nbr_t>              nbr_vec_t;

class CovModel;
class IsometricCovModel;
class NoiseModel;
class DistFunc;
class SeqNNGP {
 public:
  SeqNNGP(const double* _y_targets, const double* _coords, const int _d_idim,
          const int _q_ldim, const int _n_samples, const int _m_nns,
          CovModel& _cm, DistFunc& _df, NoiseModel& _nm);

  /** Nearest neighbors ranges
   * Lower part holds starting index (in nnIndx) for each node
   * Upper part holds number of elements for each node
   *
   * size : [2*n]
   */
  std::vector<int> nnIndxLU;

  /** Nearest neighbors index
   * Holds the indices of the neighbors of each node in B_mat. Allocate our own
   * memory for these
   *
   * size : [nIndx]
   */
  std::vector<int> nnIndx;

  /** Distances between neighbors
   *
   * size : [nIndx]
   */
  std::vector<double> nnDist;

  /** Reverse index.  Holds which nodes have n as a neighbor
   *
   * size : [nIndx]
   */
  std::vector<int> uIndx;

  /** Ranges for reverse index.
   *
   * size : [2*n]
   */
  std::vector<int> uIndxLU;
  /** Which neighbor is it?
   *
   * size : [nIndx]
   */
  std::vector<int> uiIndx;

  /** Lower part holds cumulative sum of squared neighbors of 1:(i-1)
   * Uppoer part holds # of neighbors squared
   *
   * size : [2*n]
   */
  std::vector<int> CIndx;

  const int d;      // Number of input dimensions
  const int q;      // Number of output dimensions (classes if > 1);
  const int n;      // Number of input locations
  const int m;      // Number of nearest neighbors
  const int nIndx;  // Total number of neighbors (DAG edges)

  inline int get_num_samples() { return n; }
  inline int get_num_neighbors() { return m; }
  inline int get_sample_dim() { return d; }
  inline int get_output_dim() { return q; }

  // Use existing memory here (allocated in python-layer)
  const Eigen::Map<const MatrixXd> y;       // [q, n] ([n, q] in python)
  const Eigen::Map<const MatrixXd> coords;  // [d, n]  ([n, d] in python)

  CovModel&   cm;  // Model for GP covariances
  NoiseModel& nm;  // Model for additional measurement noise
  DistFunc&   df;  // Model for distance function

  std::random_device rd;
  std::mt19937       gen;

  // These are mostly internal, but I'm too lazy for the moment to make them
  // private. We allocate this memory ourselves.
  std::vector<double> B_mat;  // [nIndx]
  std::vector<double> F_mat;  // [n]
  std::vector<double> Bcand;  // [nIndx]
  std::vector<double> Fcand;  // [n]
  // stacked q x mq cross-covariance matrices C_{s_i, N(s_i)} between w(s_i) and
  // w_{N(s_i)}.
  // We appear to be assuming q = 1 for now.
  std::vector<double> c_crosscov;  // [nIndx]
  // stacked ~ (m x m) covariance matrices C_{N(s_i)}.
  std::vector<double> C_cov;  // [~n*m*m]
  // stacked ~ (m x m) cross pairwise distances for N(s_i).
  std::vector<double> D_dist;  // [~n*m*m]
  VectorXd            w_vec;   // [n] Latent GP samples

  // return the additive model against which the GP is modeling discrepency.
  // This is the zero vector for the raw NNGP.
  virtual VectorXd additiveModel() const { return VectorXd::Zero(n); }

  virtual void sample(int nSamples);  // One Gibbs iteration

  // Use a particular covariance model to update given B and F vectors.
  // void         updateBF(double*, double*, IsometricCovModel&);
  void         updateBF(double*, double*, CovModel&);
  virtual void updateW();

  Eigen::MatrixXd get_regression_coeffs();

  //   Eigen::MatrixXd predict() const;
  Eigen::MatrixXd predict(const Eigen::Ref<const Eigen::MatrixXd>& Xstar,
                          const int nSamples, const int epochSize,
                          const int burnin);

  /**
   * Produce maximium a posteriori (MAP) estimate for each set of input
   * coordinates. Operates over the columns of Xstar, a [dstar, nstar] matrix
   * representing nstar points of dstar coordinates. dstar must equal d.
   * Implements Eq. (5) from [2], where B_mat and F_mat correspond to A and D in
   * [2].
   */
  Eigen::MatrixXd MAPPredict(const Eigen::Ref<const Eigen::MatrixXd>& Xstar);

  /**
   * Compute a quadratic form u^T C^{-1} v in terms of B_mat and F_mat.
   * Implements Eq. (5) from [2].
   */
  double quadratic_form(const Eigen::Ref<const Eigen::VectorXd>&,
                        const Eigen::Ref<const Eigen::VectorXd>&) const;

 protected:
  void mkUIndx();
  void mkUIIndx();
  void mkCD();
  void updateWparts(const int, double&, double&, double&) const;
  void predictYstarPartsSupport(const nbr_vec_t&, double&, double&,
                                double&) const;
  void predictYstarPartsInterpolation(const nbr_vec_t&, double&, double&);
  void sampleYstar(double&, double&, const double, const double, const int,
                   const int, const int);

  inline double sparse_kernel_apply(const nbr_map_t&, const int&) const;

  //   Eigen::VectorXd regression_univariate(const Eigen::VectorXd&) const;
  Eigen::VectorXd regression_univariate(const fpq_t&) const;
  Eigen::VectorXd dense_crosscov(
      const Eigen::Ref<const Eigen::VectorXd>&) const;

  fpq_t sparse_crosscov(const Eigen::Ref<const Eigen::VectorXd>&) const;

  /**
   * Initialize regression_coeffs, a q x n matrix whose rows are of the form of
   * the vectors v from Eq. (5) of [2]. If q > 1, each row of
   * regression_coeffs corresponds to a separate regression target for the qth
   * classification label.
   *
   * We precomput regression_coeffs so as to reduce computation for the
   * regression of more than one target.
   *
   * Requires significant storage, so might not ultimately be worthwhile.
   */
  void regression_init();

  // Has regression_init() run with current B_mat, F_mat?
  bool regression_ready;

  Eigen::MatrixXd regression_coeffs;  // [q, n] ([n, q] in python)
};

}  // namespace pyNNGP

#endif
