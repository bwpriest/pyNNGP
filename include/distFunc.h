#ifndef NNGP_distFunc_h
#define NNGP_distFunc_h

#include <Eigen/Dense>
#include <cmath>
#include <functional>

namespace pyNNGP {

class DistFunc {
 public:
  DistFunc(const int id) : identity(id) {}
  virtual double operator()(const Eigen::VectorXd&,
                            const Eigen::VectorXd&) const = 0;

  const double identity;

};  // namespace pyNNGP

class EuclideanDistFunc : public DistFunc {
 public:
  EuclideanDistFunc() : DistFunc(0.0) {}
  double operator()(const Eigen::VectorXd& a,
                    const Eigen::VectorXd& b) const override {
    return (a - b).norm();
  }
};

class DotProductDistFunc : public DistFunc {
 public:
  DotProductDistFunc() : DistFunc(1.0) {}
  double operator()(const Eigen::VectorXd& a,
                    const Eigen::VectorXd& b) const override {
    assert(a.size() == b.size());
    return a.dot(b);
  }
};

}  // namespace pyNNGP

#endif
