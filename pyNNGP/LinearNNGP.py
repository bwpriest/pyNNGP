from . import _pyNNGP
import numpy as np


class LinearNNGP:
    """Linear Spatial Hierarchical Nearest Neighgor Gaussian Process (LinearNNGP)

    The implementation is based upon Datta, Banerjee, Finley, and Gelfand,
    "Hierarchical Nearest-Neighbor Guassian Process Models for Large
    Geostatistical Datasets" (https://arxiv.org/abs/1406.7343).

    We make the following assumtions:
        - the fixed design matrix X from Eq. (8) of [DBFG16] is the identity
          matrix.

    Parameters
    ----------
    y : array-like
        The target values observed at the training samples.
    X : array-like, shape = (n_samples, n_features)
        The fixed spatially-reference predictors of the linear model.
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
        X,
        coords,
        nNeighbors,
        covModel,
        distFunc,
        noiseModel,
        normalize=False,
    ):
        self.X = np.ascontiguousarray(np.atleast_2d(X))
        self.y = np.ascontiguousarray(np.atleast_2d(y))
        # Sort by coords[:, 0] first?
        self.coords = np.ascontiguousarray(np.atleast_2d(coords))
        self.nNeighbors = nNeighbors
        self.covModel = covModel
        self.distFunc = distFunc
        self.noiseModel = noiseModel

        if normalize == True:
            self.coords = self.coords / self.coords.sum(axis=0)

        if self.y.shape[0] == 1:
            self.y = np.ascontiguousarray(np.reshape(y, (self.y.shape[1], 1)))

        assert self.coords.shape[0] == self.X.shape[0] == self.y.shape[0]
        self._LinearNNGP = _pyNNGP.LinearNNGP(
            self.y.ctypes.data,  # target values
            self.X.ctypes.data,  # fixed spatially-referenced predictors
            self.coords.ctypes.data,  # input locations
            self.coords.shape[1],  # # of input dimensions
            self.y.shape[1],  # 1 or # of class labels
            self.X.shape[1],  # # of indicators per input location
            self.coords.shape[0],  #  # of location/target pairs
            self.nNeighbors,  # maximum # of nearest neighbors for conditioning
            self.covModel,  # covariance function to be used
            self.distFunc,  # distance/similarity function to be used
            self.noiseModel,  # noise prior
        )

    def sample(self, N):
        """Sample N functions from LinearNNGP and evaluate at training points.

        Parameters
        ----------
        N : int
            Number of samples to return
        """
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
