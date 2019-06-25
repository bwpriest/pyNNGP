#include "covModel.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace pyNNGP {
void pyExportCovModel(py::module& m) {
  py::class_<CovModel>(m, "CovModel");
  py::class_<IsotropicCovModel, CovModel>(m, "Isometric")
      .def_property_readonly("phi", &IsotropicCovModel::getPhi)
      .def_property_readonly("sigmaSq", &IsotropicCovModel::getSigmaSq);

  py::class_<ExponentialCovModel, IsotropicCovModel>(m, "Exponential")
      .def(py::init<double, double, double, double, double, double, double>(),
           "sigmaSq"_a, "phi"_a, "phiUnifa"_a, "phiUnifb"_a, "phiTuning"_a,
           "sigmaSqIGa"_a, "sigmaSqIGb"_a)
      .def("cov", &ExponentialCovModel::cov);

  py::class_<SphericalCovModel, IsotropicCovModel>(m, "Spherical")
      .def(py::init<double, double, double, double, double, double, double>(),
           "sigmaSq"_a, "phi"_a, "phiUnifa"_a, "phiUnifb"_a, "phiTuning"_a,
           "sigmaSqIGa"_a, "sigmaSqIGb"_a)
      .def("cov", &SphericalCovModel::cov);

  py::class_<SqExpCovModel, IsotropicCovModel, CovModel>(m, "SqExp")
      .def(py::init<double, double, double, double, double, double, double>(),
           "sigmaSq"_a, "phi"_a, "phiUnifa"_a, "phiUnifb"_a, "phiTuning"_a,
           "sigmaSqIGa"_a, "sigmaSqIGb"_a)
      .def("cov", &SqExpCovModel::cov);
}
}  // namespace pyNNGP
