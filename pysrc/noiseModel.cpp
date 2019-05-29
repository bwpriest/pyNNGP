#include "noiseModel.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace pyNNGP {
static ConstHeterogeneousNoiseModel* MakeCHNM(size_t it, int n) {
  const double* tauSq = reinterpret_cast<double*>(it);

  return new ConstHeterogeneousNoiseModel(tauSq, n);
}

void pyExportNoiseModel(py::module& m) {
  py::class_<NoiseModel>(m, "NoiseModel");

  py::class_<IGNoiseModel, NoiseModel>(m, "IGNoiseModel")
      .def(py::init<double, double, double>());

  py::class_<ConstHomogeneousNoiseModel, NoiseModel>(
      m, "ConstHomogeneousNoiseModel")
      .def(py::init<double>(), "tauSq"_a);

  py::class_<ConstHeterogeneousNoiseModel, NoiseModel>(
      m, "ConstHeterogeneousNoiseModel")
      .def(py::init(&MakeCHNM));
}
}  // namespace pyNNGP
