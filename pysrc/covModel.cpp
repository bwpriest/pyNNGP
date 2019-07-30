#include "covModel.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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

  py::class_<NeuralNetworkCovModel, CovModel>(m, "NNKernel")
      .def(py::init<int, double, double>(), "L"_a, "sigmaSqW"_a, "sigmaSqB"_a)
      .def_property_readonly("sigmaSqW", &NeuralNetworkCovModel::getSigmaSqW)
      .def_property_readonly("sigmaSqB", &NeuralNetworkCovModel::getSigmaSqB)
      .def_property_readonly("Kxx", &NeuralNetworkCovModel::getKxx)
      .def("cov", &NeuralNetworkCovModel::cov);
}
}  // namespace pyNNGP
