from . import _pyNNGP
import numpy as np


class SeqNNGP:
    def __init__(self, y, coords, nNeighbors, covModel, distFunc, noiseModel):
        self.y = np.ascontiguousarray(np.atleast_1d(y))
        # Sort by coords[:, 0] first?
        self.coords = np.ascontiguousarray(np.atleast_2d(coords))
        self.nNeighbors = nNeighbors
        self.covModel = covModel
        self.distFunc = distFunc
        self.noiseModel = noiseModel

        self._SeqNNGP = _pyNNGP.SeqNNGP(
            self.y.ctypes.data,  # target values
            self.coords.ctypes.data,  # input locations
            self.coords.shape[1],  #
            self.coords.shape[0],
            self.nNeighbors,
            self.covModel,
            self.distFunc,
            self.noiseModel,
        )

    def sample(self, N):
        self._SeqNNGP.sample(N)

    def updateW(self):
        self._SeqNNGP.updateW()

    def updateTauSq(self):
        self._SeqNNGP.updateTauSq()

    @property
    def w(self):
        return self._SeqNNGP.w

    @property
    def tauSq(self):
        return self._SeqNNGP.tauSq

    @property
    def B(self):
        return self._SeqNNGP.B

    @property
    def F(self):
        return self._SeqNNGP.F
