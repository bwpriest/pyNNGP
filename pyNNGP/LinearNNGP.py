from . import _pyNNGP
import numpy as np


class LinearNNGP:
    def __init__(
        self, y, X, coords, nNeighbors, covModel, distFunc, noiseModel
    ):
        self.X = np.ascontiguousarray(np.atleast_2d(X))
        self.y = np.ascontiguousarray(np.atleast_1d(y))
        # Sort by coords[:, 0] first?
        self.coords = np.ascontiguousarray(np.atleast_2d(coords))
        self.nNeighbors = nNeighbors
        self.covModel = covModel
        self.distFunc = distFunc
        self.noiseModel = noiseModel

        assert self.coords.shape[0] == self.X.shape[0] == self.y.shape[0]
        self._LinearNNGP = _pyNNGP.LinearNNGP(
            self.y.ctypes.data,
            self.X.ctypes.data,
            self.coords.ctypes.data,
            self.coords.shape[1],
            self.X.shape[1],
            self.coords.shape[0],
            self.nNeighbors,
            self.covModel,
            self.distFunc,
            self.noiseModel,
        )

    def sample(self, N):
        self._LinearNNGP.sample(N)

    def updateW(self):
        self._LinearNNGP.updateW()

    def updateBeta(self):
        self._LinearNNGP.updateBeta()

    def updateTauSq(self):
        self._LinearNNGP.updateTauSq()

    @property
    def w(self):
        return self._LinearNNGP.w

    @property
    def beta(self):
        return self._LinearNNGP.beta

    @property
    def tauSq(self):
        return self._LinearNNGP.tauSq

    @property
    def B(self):
        return self._LinearNNGP.B

    @property
    def F(self):
        return self._LinearNNGP.F
