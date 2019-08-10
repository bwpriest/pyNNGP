#include "LinearNNGP.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include "LinearNNGP.h"
#include "covModel.h"
#include "distFunc.h"
#include "noiseModel.h"

namespace py = pybind11;

namespace pyNNGP {
static LinearNNGP* MakeLinearNNGP(size_t iy, size_t iX, size_t icoords, int d,
                                  int q, int p, int n, int nNeighbors,
                                  CovModel& cm, DistFunc& df, CompFunc& cf, 
                                  NoiseModel& nm) {
  const double* y      = reinterpret_cast<double*>(iy);
  const double* X      = reinterpret_cast<double*>(iX);
  const double* coords = reinterpret_cast<double*>(icoords);

  return new LinearNNGP(y, X, coords, d, q, p, n, nNeighbors, cm, df, cf, nm);
}

void pyExportLinearNNGP(py::module& m) {
  py::class_<LinearNNGP>(m, "LinearNNGP")
      .def(py::init(&MakeLinearNNGP))
      .def("sample", &LinearNNGP::sample)
      .def("updateW", &LinearNNGP::updateW)
      .def("updateBeta", &LinearNNGP::updateBeta)
      .def_property_readonly(
          "nnIndx",
          [](LinearNNGP& s) -> py::array_t<int> {
            return {
                {s.nnIndx.size()}, {sizeof(int)}, &s.nnIndx[0], py::cast(s)};
          })
      .def_property_readonly("nnIndxLU",
                             [](LinearNNGP& s) -> py::array_t<int> {
                               return {{s.nnIndxLU.size()},
                                       {sizeof(int)},
                                       &s.nnIndxLU[0],
                                       py::cast(s)};
                             })
      .def_property_readonly(
          "nnDist",
          [](LinearNNGP& s) -> py::array_t<double> {
            return {
                {s.nnDist.size()}, {sizeof(double)}, &s.nnDist[0], py::cast(s)};
          })
      .def_property_readonly(
          "uIndx",
          [](LinearNNGP& s) -> py::array_t<int> {
            return {{s.uIndx.size()}, {sizeof(int)}, &s.uIndx[0], py::cast(s)};
          })
      .def_property_readonly(
          "uIndxLU",
          [](LinearNNGP& s) -> py::array_t<int> {
            return {
                {s.uIndxLU.size()}, {sizeof(int)}, &s.uIndxLU[0], py::cast(s)};
          })
      .def_property_readonly(
          "uiIndx",
          [](LinearNNGP& s) -> py::array_t<int> {
            return {
                {s.uiIndx.size()}, {sizeof(int)}, &s.uiIndx[0], py::cast(s)};
          })
      .def_property_readonly(
          "CIndx",
          [](LinearNNGP& s) -> py::array_t<int> {
            return {{s.CIndx.size()}, {sizeof(int)}, &s.CIndx[0], py::cast(s)};
          })
      .def_property_readonly(
          "B",
          [](LinearNNGP& s) -> py::array_t<double> {
            return {
                {s.B_mat.size()}, {sizeof(double)}, &s.B_mat[0], py::cast(s)};
          })
      .def_property_readonly(
          "F",
          [](LinearNNGP& s) -> py::array_t<double> {
            return {
                {s.F_mat.size()}, {sizeof(double)}, &s.F_mat[0], py::cast(s)};
          })
      .def_property_readonly(
          "w",
          [](LinearNNGP& s) -> py::array_t<double> {
            return {
                {s.w_vec.size()}, {sizeof(double)}, &s.w_vec[0], py::cast(s)};
          })
      .def_property_readonly("beta", [](LinearNNGP& s) -> py::array_t<double> {
        return {{s.beta.size()}, {sizeof(double)}, &s.beta[0], py::cast(s)};
      });
}
}  // namespace pyNNGP
