from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from .learn_rate import Adam


def _crps_score(mu, sigma, y):
    z = (y - mu) / sigma
    score = sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
    return score


def _natural_gradient(mu, sigma, y):
    z = (y - mu) / sigma
    dl_dmu = (1 - 2 * norm.cdf(z)) * sigma * np.sqrt(np.pi)
    dl_dlog_sigma = (2 * norm.pdf(z) - 1 / np.pi) * 2 * np.sqrt(np.pi)
    return dl_dmu, dl_dlog_sigma


@dataclass(init=False)
class _DistParams:
    # Distribution parameters
    # Location - mu
    # Scale - sigma
    mu: NDArray
    sigma: NDArray

    # Backprop related
    dl_dmu: NDArray
    dl_dlog_sigma: NDArray

    def __str__(self):
        return (
            f'μ={self.mu}\n'
            f'σ={self.sigma}\n'
        )


class NormalDist:
    def __init__(self):
        self._mu = None
        self._sigma = None
        self._log_sigma = None

        self._dl_dmu = None
        self._dl_dlog_sigma = None

        # Learning rate regulators
        self._mu_lr = Adam()
        self._log_sigma_lr = Adam()

        self._mu_hat = None
        self._sigma_hat = None
        self._pi = None

    @staticmethod
    def compute_estimate(target):
        params = _DistParams()
        params.mu = target.mean(keepdims=True)
        params.sigma = target.std(keepdims=True).clip(min=1e-2)
        return params

    @staticmethod
    def compute_score(params: _DistParams, target):
        return _crps_score(mu=params.mu, sigma=params.sigma, y=target).sum()

    def create_parameters(self, leaves):
        self._mu = np.stack([leaf.dist_params.mu for leaf in leaves])
        self._sigma = np.stack([leaf.dist_params.sigma for leaf in leaves])
        self._log_sigma = np.log(self._sigma)

        self._dl_dmu = np.zeros_like(self._mu)
        self._dl_dlog_sigma = np.zeros_like(self._log_sigma)

        for index, leaf in enumerate(leaves):
            leaf.dist_params.mu = self._mu[index, np.newaxis]
            leaf.dist_params.sigma = self._sigma[index, np.newaxis]

            leaf.dist_params.dl_dmu = self._dl_dmu[index, np.newaxis]
            leaf.dist_params.dl_dlog_sigma = self._dl_dlog_sigma[index, np.newaxis]

    def adjust_params(self):
        self._mu_lr.adjust(param=self._mu, grad=self._dl_dmu)
        self._log_sigma_lr.adjust(param=self._log_sigma, grad=self._dl_dlog_sigma)

        self._sigma[:] = np.exp(self._log_sigma)

    def forward_prop(self, leaves):
        self._pi = np.stack([leaf.pi for leaf in leaves])
        self._mu_hat = np.sum(self._pi * self._mu, axis=0)
        self._sigma_hat = np.sum(self._pi * self._sigma, axis=0)

        prediction = norm(loc=self._mu_hat, scale=self._sigma_hat)
        return prediction

    def backward_prop(self, target, leaves):
        loss = _crps_score(mu=self._mu_hat, sigma=self._sigma_hat, y=target).mean()
        dl_dmu_hat, dl_dlog_sigma_hat = _natural_gradient(
            mu=self._mu_hat, sigma=self._sigma_hat, y=target
        )

        for leaf in leaves:
            dmu_hat_dmu = leaf.pi
            dsigma_hat_dsigma = leaf.pi
            dlog_sigma_hat_dlog_sigma = dsigma_hat_dsigma * leaf.dist_params.sigma / self._sigma_hat

            dmu_hat_dpi = leaf.dist_params.mu
            dsigma_hat_dpi = leaf.dist_params.sigma
            dlog_sigma_hat_dpi = dsigma_hat_dpi / self._sigma_hat

            leaf.dl_dpi = dl_dmu_hat * dmu_hat_dpi + dl_dlog_sigma_hat * dlog_sigma_hat_dpi
            dl_dmu = dl_dmu_hat * dmu_hat_dmu
            dl_dlog_sigma = dl_dlog_sigma_hat * dlog_sigma_hat_dlog_sigma

            leaf.dist_params.dl_dmu[:] = dl_dmu.mean()
            leaf.dist_params.dl_dlog_sigma[:] = dl_dlog_sigma.mean()

        return loss
