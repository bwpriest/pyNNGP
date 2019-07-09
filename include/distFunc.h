#ifndef NNGP_distFunc_h
#define NNGP_distFunc_h

#include <Eigen/Dense>
#include <cmath>

namespace pyNNGP {

class DistFunc {
 public:
  virtual double operator()(const Eigen::VectorXd&,
                            const Eigen::VectorXd&) const = 0;
};

class EuclideanDistFunc : public DistFunc {
 public:
  double operator()(const Eigen::VectorXd& a,
                    const Eigen::VectorXd& b) const override {
    return (a - b).norm();
  }
};

class DotProductDistFunc : public DistFunc {
 public:
  double operator()(const Eigen::VectorXd& a,
                    const Eigen::VectorXd& b) const override {
    return a.dot(b);
  }
};

}  // namespace pyNNGP

#endif
