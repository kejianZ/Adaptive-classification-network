"""
CSCC11 - Introduction to Machine Learning, Fall 2021, Assignment 2
M. Ataei
"""


import numpy as np
from functools import partial
import math
from scipy.stats import multivariate_normal
class GMM:
    def __init__(self, init_centers):
        assert len(init_centers.shape) == 2, f"init_centers should be a KxD matrix. Got: {init_centers.shape}"
        (self.K, self.D) = init_centers.shape
        assert self.K > 1, f"There must be at least 2 clusters. Got: {self.K}"

        # Shape: K x D
        self.centers = np.copy(init_centers)

        # Shape: K x D x D
        self.covariances = np.tile(np.eye(self.D), reps=(self.K, 1, 1))

        # Shape: K x 1
        self.mixture_proportions = np.ones(shape=(self.K, 1)) / self.K

    def normal(self, x, mu, sigma, d):
        x = x.reshape(-1,1)
        mu = mu.reshape(-1,1)
        det = np.linalg.det(sigma)

        scalar = 1/((2*math.pi)**(d/2) * det**(1/2))

        res = x-mu
        inv = np.linalg.inv(sigma)
        power = -1/2 * res.T @ inv @ res
        return scalar * np.exp(power[0][0])

    def predict_proba(self, X, phi):
        likelihood = np.zeros((X.shape[0], self.K))
        for i in range(self.K):
            distribution = multivariate_normal(
                mean=self.centers[i],
                cov=self.covariances[i])
            likelihood[:, i] = distribution.pdf(X)

        numerator = likelihood * phi.T
        denominator = numerator.sum(axis=1)[:, np.newaxis]
        weights = numerator / denominator
        return weights

    def _e_step(self, train_X):
        (N, D) = train_X.shape
        probability_matrix = np.empty(shape=(N, self.K))
        probability_matrix = self.predict_proba(train_X, self.mixture_proportions)

        assert probability_matrix.shape == (train_X.shape[0], self.K), f"probability_matrix shape mismatch. Expected: {(train_X.shape[0], self.K)}. Got: {probability_matrix.shape}"

        return probability_matrix


    def _m_step(self, train_X, probability_matrix):
        (N, D) = train_X.shape

        centers = np.empty(shape=(self.K, self.D))
        covariances = np.empty(shape=(self.K, self.D, self.D))
        mixture_proportions = np.empty(shape=(self.K, 1))
        mixture_proportions = probability_matrix.mean(axis=0).reshape(-1,1)
        for i in range(self.K):
            weight = probability_matrix[:, [i]]
            total_weight = weight.sum()
            centers[i] = (train_X * weight).sum(axis=0) / total_weight
            covariances[i] = np.cov(train_X.T,
                                   aweights=(weight / total_weight).flatten(),
                                   bias=True)

        assert centers.shape == (self.K, self.D), f"centers shape mismatch. Expected: {(self.K, self.D)}. Got: {centers.shape}"
        assert covariances.shape == (self.K, self.D, self.D), f"covariances shape mismatch. Expected: {(self.K, self.D, self.D)}. Got: {covariances.shape}"
        assert mixture_proportions.shape == (self.K, 1), f"mixture_proportions shape mismatch. Expected: {(self.K, 1)}. Got: {mixture_proportions.shape}"

        return centers, covariances, mixture_proportions

    def train(self, train_X, max_iterations=1000):
        assert len(train_X.shape) == 2 and train_X.shape[1] == self.D, f"train_X should be a NxD matrix. Got: {train_X.shape}"
        assert max_iterations > 0, f"max_iterations must be positive. Got: {max_iterations}"
        N = train_X.shape[0]

        e_step = partial(self._e_step, train_X=train_X)
        m_step = partial(self._m_step, train_X=train_X)

        labels = np.empty(shape=(N, 1), dtype=np.long)
        for _ in range(max_iterations):
            old_labels = labels.copy()
            # E-Step
            probability_matrix = e_step()

            # Reassign labels
            labels = np.argmax(probability_matrix, axis=1).reshape((N, 1))

            # Check convergence
            if np.allclose(old_labels, labels):
                break

            # M-Step
            self.centers, self.covariances, self.mixture_proportions = m_step(probability_matrix=probability_matrix)

        return labels

