#include "SeqNNGP.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include "covModel.h"
#include "distFunc.h"
#include "noiseModel.h"

namespace py = pybind11;

namespace pyNNGP {
static SeqNNGP* MakeSeqNNGP(size_t iy, size_t icoords, int d, int n,
                            int nNeighbors, CovModel& cm, DistFunc& df,
                            NoiseModel& nm) {
  const double* y      = reinterpret_cast<double*>(iy);
  const double* coords = reinterpret_cast<double*>(icoords);

  return new SeqNNGP(y, coords, d, n, nNeighbors, cm, df, nm);
}

void pyExportSeqNNGP(py::module& m) {
  py::class_<SeqNNGP>(m, "SeqNNGP")
      .def(py::init(&MakeSeqNNGP))
      .def("sample", &SeqNNGP::sample)
      .def("updateW", &SeqNNGP::updateW)
      .def_property_readonly(
          "nnIndx",
          [](SeqNNGP& s) -> py::array_t<int> {
            return {
                {s.nnIndx.size()}, {sizeof(int)}, &s.nnIndx[0], py::cast(s)};
          })
      .def_property_readonly("nnIndxLU",
                             [](SeqNNGP& s) -> py::array_t<int> {
                               return {{s.nnIndxLU.size()},
                                       {sizeof(int)},
                                       &s.nnIndxLU[0],
                                       py::cast(s)};
                             })
      .def_property_readonly(
          "nnDist",
          [](SeqNNGP& s) -> py::array_t<double> {
            return {
                {s.nnDist.size()}, {sizeof(double)}, &s.nnDist[0], py::cast(s)};
          })
      .def_property_readonly(
          "uIndx",
          [](SeqNNGP& s) -> py::array_t<int> {
            return {{s.uIndx.size()}, {sizeof(int)}, &s.uIndx[0], py::cast(s)};
          })
      .def_property_readonly(
          "uIndxLU",
          [](SeqNNGP& s) -> py::array_t<int> {
            return {
                {s.uIndxLU.size()}, {sizeof(int)}, &s.uIndxLU[0], py::cast(s)};
          })
      .def_property_readonly(
          "uiIndx",
          [](SeqNNGP& s) -> py::array_t<int> {
            return {
                {s.uiIndx.size()}, {sizeof(int)}, &s.uiIndx[0], py::cast(s)};
          })
      .def_property_readonly(
          "CIndx",
          [](SeqNNGP& s) -> py::array_t<int> {
            return {{s.CIndx.size()}, {sizeof(int)}, &s.CIndx[0], py::cast(s)};
          })
      .def_property_readonly(
          "B",
          [](SeqNNGP& s) -> py::array_t<double> {
            return {{s.B.size()}, {sizeof(double)}, &s.B[0], py::cast(s)};
          })
      .def_property_readonly(
          "F",
          [](SeqNNGP& s) -> py::array_t<double> {
            return {{s.F.size()}, {sizeof(double)}, &s.F[0], py::cast(s)};
          })
      .def_property_readonly("w", [](SeqNNGP& s) -> py::array_t<double> {
        return {{s.w.size()}, {sizeof(double)}, &s.w[0], py::cast(s)};
      });
}
}  // namespace pyNNGP
