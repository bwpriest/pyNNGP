#include "distFunc.h"
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace pyNNGP {
void pyExportDistFunc(py::module& m) {
  py::class_<DistFunc>(m, "DistFunc");

  py::class_<EuclideanDistFunc, DistFunc>(m, "Euclidean")
      .def(py::init<>())
      .def("__call__", [](const EuclideanDistFunc& f, const Eigen::VectorXd& a,
                          const Eigen::VectorXd& b) { return f(a, b); });
  py::class_<DotProductDistFunc, DistFunc>(m, "DotProduct")
      .def(py::init<>())
      .def("__call__", [](const DotProductDistFunc& f, const Eigen::VectorXd& a,
                          const Eigen::VectorXd& b) { return f(a, b); });
}
}  // namespace pyNNGP
