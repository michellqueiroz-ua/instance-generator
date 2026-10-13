"""Minimal scipy.stats replacement for the parts of ``fitter`` the legacy
scripts used (fitter is GPL-3.0; REQreate is MIT).

Same method as fitter: fit each candidate distribution by maximum likelihood,
then rank by the sum of squared errors between its pdf and the data histogram.
"""

import warnings

import numpy as np
import pandas as pd
from scipy import stats

COMMON_DISTRIBUTIONS = [
    "cauchy", "chi2", "expon", "exponpow", "gamma",
    "lognorm", "norm", "powerlaw", "rayleigh", "uniform",
]


def get_common_distributions():
    return list(COMMON_DISTRIBUTIONS)


def get_distributions():
    return [name for name in dir(stats)
            if isinstance(getattr(stats, name), stats.rv_continuous)]


class Fitter:

    def __init__(self, data, timeout=None, distributions=None, bins=100):
        # timeout is accepted for call compatibility; fits run to completion.
        self.data = np.asarray(data, dtype=float).ravel()
        self.data = self.data[np.isfinite(self.data)]
        self.distributions = distributions or get_common_distributions()
        self.bins = bins
        self.fitted_param = {}
        self.df_errors = pd.DataFrame()

    def fit(self):
        density, edges = np.histogram(self.data, bins=self.bins, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        rows = []
        for name in self.distributions:
            dist = getattr(stats, name)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    params = dist.fit(self.data)
                    pdf = dist.pdf(centers, *params)
                sse = float(np.sum((pdf - density) ** 2))
            except Exception:
                continue
            if not np.isfinite(sse):
                continue
            self.fitted_param[name] = params
            rows.append({"distribution": name, "sumsquare_error": sse})
        self.df_errors = (pd.DataFrame(rows).set_index("distribution")
                          .sort_values("sumsquare_error"))
        return self

    def summary(self, Nbest=5, plot=False):
        return self.df_errors.head(Nbest)

    def get_best(self, method="sumsquare_error"):
        name = self.df_errors[method].idxmin()
        return {name: self.fitted_param[name]}
