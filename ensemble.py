from sklearn.decomposition import PCA
from scipy.linalg import svd
from copy import deepcopy
from typing import List
import numpy as np

np.warnings.filterwarnings('ignore')


class HashBasedUndersamplingEnsemble:
    RECIPROCAL, RANDOM, LINEARITY, NEGATIVE_EXPONENT, LIMIT = [
        'reciprocal',
        'random',
        'linearity',
        'negexp',
        'limit'
    ]

    SUPPORTED_SMPLINGS = [
        RECIPROCAL, RANDOM, LINEARITY, NEGATIVE_EXPONENT, LIMIT
    ]

    def __init__(
            self,
            base_estimator,
            sampling: str = RECIPROCAL,
            n_iterations: int = 50,
            random_state: int = None,
            normalization: bool = False,
    ):
        """
        Hashing-Based Undersampling Ensemble for Imbalanced Pattern Classification Problems

        :param base_estimator:
            Base Estimator

        :param sampling: str (default = 'normal')
            sampling method
            supported methods: 'reciprocal', 'normal', 'random', 'linearity', 'negexp', 'limit'

        :param n_iterations: int (default = 50)
            maximum iteration for Iterative Quantization

        :param random_state: int (default = None)
            random state for Iterative Quantization

        :param normalization: bool (default = False)
            normalize the data
        """
        self.base_estimator = base_estimator

        self.sampling: str = sampling
        if self.sampling not in self.SUPPORTED_SMPLINGS:
            raise ValueError('supported sampling: {}'.format(
                self.SUPPORTED_SMPLINGS
            ))

        self.n_iterations: int = n_iterations
        if type(self.n_iterations) != int \
                or not (0 < self.n_iterations):
            raise ValueError('n_iterations should be an integer number bigger than 0')

        self.random_state: int = random_state
        np.random.seed(self.random_state)

        # store classifiers
        self.classifiers: List = list()

        # normalization condition
        self.normalization: bool = normalization

    def _check_fitted(self):
        assert self._is_fitted, 'fit function not called yet'

    def _check_Xy(self, X, y: np.array = None) -> [np.array, np.array]:
        """Check X and y to be valid"""

        if len(X.shape) != 2:
            raise ValueError('X should be 2D (n_samples x n_features)')

        if y is not None:
            n_samples, n_features = X.shape
            if len(y.flatten()) != n_samples:
                raise ValueError('number of samples in y is not equal to X')

            self.classes_, self.n_classes_ = np.unique(y, return_counts=True)
            if len(self.classes_) > 2:
                raise NotImplementedError('Just binary class supported'
                                          ', multi class not supported yet')

        if self.normalization:
            # Centered Data (normalization step)
            X = X - np.mean(X, axis=0)

        return X, y

    def _itq(self, X: np.array):
        """Iterative Quantitization

        :param X: np.array (n_sample, n_features)
            projected feature matrix via PCA

        :return R: np.array
            rotate matrix
        """

        # Construct Orthogonal rotation matrix
        R = np.random.randn(self.n_bits, self.n_bits)
        [U, _, _] = svd(R)
        R = U[:, :self.n_bits]

        # Find Optimal Rotation
        for _ in range(self.n_iterations):
            V = X @ R
            [U, _, VT] = svd(np.sign(V).T @ X)
            R = (VT @ U.T)

        return R

    def _sampling(self, X: np.array, subspace: np.array, nMin: int):
        """Sampling Methods
        1. Reciprocal
        2. All Random
        3. Linearity
        4. Negative Exponent
        5. Limit
        """
        # get number of samples
        n_samples, _ = X.shape

        # Calculate Hamming Distance for all sample
        distance = np.sum(
            np.unpackbits(X ^ subspace, axis=1, count=self.n_bits, bitorder='little')
            , axis=1
        )

        if self.sampling == self.RANDOM:
            """All Random"""
            # weights = np.ones((n_samples,)) * 0.5
            return np.random.choice(n_samples, nMin)

        elif self.sampling == self.LINEARITY:
            """Linearity"""

            # calculate weights
            weights = (self.n_bits + 1 - distance) / (self.n_bits + 1)

        elif self.sampling == self.NEGATIVE_EXPONENT:
            """Negative Exponent"""

            # calculate weights
            weights = 1 / (np.power(2, distance))

        elif self.sampling == self.LIMIT:
            """Limit"""

            # calculate weights
            weights = np.where(distance == 0, 1, 0)

        else:
            """Reciprocal"""

            # calculate weights
            weights = np.nan_to_num(
                1 / (distance * np.power(2, self.n_bits))
                , nan=1, neginf=1, posinf=1
            )

        # Shuffle weights for sampling (we can make the sampling randomness
        # for selecting from the surronding subspaces)
        np.random.shuffle(weights)

        # Sort weights by their weights and so Pick Nmin samples due to weight
        # distribution to form the training subset
        return np.argsort(weights)[::-1][:nMin]

    def fit(self, X: np.array, y: np.array):
        """Fitting Function

        X: np.array (n_samples, n_features)
            features matrix

        y: np.array (n_samples,)
            labels vector
        """
        # Validate X and y
        X, y = self._check_Xy(X, y)

        # TODO###############################[REFACTORE NEEDED]################################
        argsort = np.argsort(self.n_classes_)
        Cmin, Cmaj = self.classes_[argsort]
        Nmin, Nmaj = self.n_classes_[argsort]

        min_indexes = np.where(y != Cmaj)[0]
        maj_indexes = np.where(y == Cmaj)[0]

        Xmaj, ymaj = X[maj_indexes], y[maj_indexes]
        # TODO##################################################################################

        # Get number of Bits
        self.n_bits = np.min([
            np.ceil(np.log2(3 * Nmaj / Nmin)).astype(np.int),
            *X.shape
        ])

        # Using PCA (Dimension Reduction)
        self.pca = PCA(n_components=self.n_bits)

        # Transform X
        V = self.pca.fit_transform(Xmaj)

        # Using Iterative Quantitization (Rotation Matrix)
        self.R = self._itq(V)

        # V x R
        U = np.sign(V @ self.R).astype(np.int)

        # Assign each sample to Hash Code Subspace
        Q = np.packbits(np.where(U < 0, 0, U), axis=1, bitorder='little')

        for subspace in range(np.power(2, self.n_bits)):

            # Pick Nmin samples due to weight distribution w to form the training subset
            selected = self._sampling(Q, subspace, Nmin)

            # Train base classifier C using T and minority samples
            C = deepcopy(self.base_estimator)

            # TODO: refactor it
            X_ = np.concatenate((X[min_indexes], Xmaj[selected]))
            y_ = np.concatenate((y[min_indexes], ymaj[selected]))

            # store all classifiers for prediction step
            self.classifiers.append(C.fit(X_, y_))

        self._is_fitted = True
        return self

    def predict(self, X: np.array):
        """Prediction Function"""
        self._check_fitted()

        # Check and normalize X
        X, _ = self._check_Xy(X)

        # Prediction step
        H = np.sum([
            classifier.predict(X) for classifier in self.classifiers
        ], axis=0)

        # Apply sign function over result of classifiers
        return np.sign(H)
