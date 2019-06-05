#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace pyNNGP {
void pyExportSeqNNGP(py::module& m);
void pyExportLinearNNGP(py::module& m);
void pyExportCovModel(py::module& m);
void pyExportNoiseModel(py::module& m);
void pyExportDistFunc(py::module& m);

PYBIND11_MODULE(_pyNNGP, m) {
  pyExportSeqNNGP(m);
  pyExportLinearNNGP(m);
  pyExportCovModel(m);
  pyExportNoiseModel(m);
  pyExportDistFunc(m);
}
}  // namespace pyNNGP
