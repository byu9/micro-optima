import numpy as np


class Adam:
    _delta = 1e-8

    __slots__ = (
        '_epsilon',
        '_rho1',
        '_rho2',
        '_s',
        '_r',
        '_t',
    )

    def __init__(self, step_size=1.0e-2, decay1=0.9, decay2=0.999):
        assert step_size > 0.0
        assert 0.0 <= decay1 < 1.0
        assert 0.0 <= decay2 < 1.0

        self._epsilon = step_size
        self._rho1 = decay1
        self._rho2 = decay2
        self._s = 0.0
        self._r = 0.0
        self._t = 0

    def adjust(self, param, grad):
        assert param.shape == grad.shape

        self._t += 1
        self._s = self._rho1 * self._s + (1 - self._rho1) * grad
        self._r = self._rho2 * self._r + (1 - self._rho2) * np.square(grad)

        shat = self._s / (1 - np.power(self._rho1, self._t))
        rhat = self._r / (1 - np.power(self._rho2, self._t))

        param -= self._epsilon * shat / (np.sqrt(rhat) + self._delta)

    def reset(self):
        self._s = 0.0
        self._r = 0.0
        self._t = 0
