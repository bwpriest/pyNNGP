from . import _pyNNGP
import numpy as np


class SeqNNGP:
    """Nearest Neighgor Gaussian Process (NNGP)

    The implementation is based upon Datta, Banerjee, Finley, and Gelfand,
    "Hierarchical Nearest-Neighbor Guassian Process Models for Large
    Geostatistical Datasets" (https://arxiv.org/abs/1406.7343).

    Parameters
    ----------
    y : array-like
        The target values observed at the training samples.
    coords : array-like, shape = (n_samples, n_features)
        The coordinate tuples corresponding to the observed targets.
    nNeighbors : int
        The maximum number of nearest neighbors upon which sample likelihood
        functions may be conditioned.
    covModel : CovModel object
        The kernel specifying the covariance function of the GP.
    distFunc : DistFunc object
        The function determining how nearest neighbors will be assigned.
    noiseModel : NoiseModel object
        The model specifying priors on the model noise.
    normalize : boolean, optional (default: False)
        Flag indicating whether to transform coords to be row-stochastic.
    """

    def __init__(
        self,
        y,
        coords,
        nNeighbors,
        covModel,
        distFunc,
        noiseModel,
        normalize=False,
    ):
        self.y = np.ascontiguousarray(np.atleast_1d(y))
        # Sort by coords[:, 0] first?
        self.coords = np.ascontiguousarray(np.atleast_2d(coords))
        self.nNeighbors = nNeighbors
        self.covModel = covModel
        self.distFunc = distFunc
        self.noiseModel = noiseModel

        if normalize == True:
            self.coords = self.coords / self.coords.sum(axis=0)

        assert self.coords.shape[0] == self.y.shape[0]
        self._SeqNNGP = _pyNNGP.SeqNNGP(
            self.y.ctypes.data,  # target values
            self.coords.ctypes.data,  # input features
            self.coords.shape[1],  # # of input of dimensions
            self.coords.shape[0],  # # of sample/target pairs
            self.nNeighbors,  # maximum # of nearest neighbors for conditioning
            self.covModel,  # covariance function to be used
            self.distFunc,  # distance/similarity function to be used
            self.noiseModel,  # noise prior
        )

    def sample(self, N):
        """Sample N functions from NNGP and evaluate at training points.

        Parameters
        ----------
        N : int
            Number of samples to return
        """
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
