// #pragma once

#ifndef NNGP_distFunc_h
#define NNGP_distFunc_h

#include <Eigen/Dense>
#include <cmath>
#include <functional>

namespace pyNNGP {

class DistFunc {
 public:
  DistFunc(const int id) : identity(id) {}
  virtual double operator()(const Eigen::VectorXd &,
                            const Eigen::VectorXd &) const = 0;

  const double identity;
};  // namespace pyNNGP

class EuclideanDistFunc : public DistFunc {
 public:
  EuclideanDistFunc() : DistFunc(0.0) {}
  double operator()(const Eigen::VectorXd &a,
                    const Eigen::VectorXd &b) const override {
    return (a - b).norm();
  }
};

class DotProductDistFunc : public DistFunc {
 public:
  DotProductDistFunc() : DistFunc(1.0) {}
  double operator()(const Eigen::VectorXd &a,
                    const Eigen::VectorXd &b) const override {
    assert(a.size() == b.size());
    return a.dot(b);
  }
};

class CompFunc {
 public:
  virtual bool operator()(const double &lhs, const double &rhs) const = 0;
};

class LessCompFunc : public CompFunc {
 public:
  template <typename T>
  bool operator()(const double &lhs, const double &rhs) const {
    return lhs < rhs;
  }
};

class GreaterCompFunc : public CompFunc {
 public:
  template <typename T>
  bool operator()(const double &lhs, const double &rhs) const {
    return lhs > rhs;
  }
};

}  // namespace pyNNGP

#endif
