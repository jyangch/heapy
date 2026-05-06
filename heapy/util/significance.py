"""Compute Poisson and Gaussian statistical significance for count measurements.

Provides scalar and vectorised implementations of ``x * log(y)`` with
safe handling of the ``0 * log(0)`` edge case, a broadcast helper, and
two significance functions: ``pgsig`` (Gaussian background) and ``ppsig``
(Poisson background with optional systematic uncertainties).

Note:
    This module is substantially derived from ``gv_significance``
    (https://github.com/giacomov/gv_significance.git).
    Reference: Giacomo Vianello, 2018, ApJS,
    https://doi.org/10.3847/1538-4365/aab780.
    BSD 3-Clause License. Copyright (c) 2018, Giacomo Vianello.
    All rights reserved.
"""

from math import log

import numpy as np
from numpy import sqrt, squeeze
import scipy.optimize


def xlogy(x, y):
    """Compute ``x * log(y)`` returning 0 when both ``x`` and ``y`` are 0.

    Avoids ``nan`` or ``inf`` that would arise from ``0 * log(0)`` under
    standard floating-point arithmetic.

    Args:
        x: Scalar multiplier. Must be a number, not an array.
        y: Scalar argument of the logarithm. Must be a number, not an array.

    Returns:
        ``0.0`` when ``x`` is ``0.0``; otherwise ``x * log(y)``.
    """

    if x == 0.0:
        return 0.0

    else:
        return x * log(y)


def xlogyv(x, y):
    """Compute ``x * log(y)`` element-wise, returning 0 where ``x`` is 0.

    Array-valued counterpart of ``xlogy``. Avoids ``nan`` or ``inf`` that
    would arise from ``0 * log(0)`` under standard floating-point arithmetic.

    Args:
        x: Multiplier array (or scalar). Converted to ``numpy.ndarray`` internally.
        y: Argument of the logarithm (or scalar). Converted to ``numpy.ndarray``
            internally; must broadcast with ``x``.

    Returns:
        Array of the same shape as ``y`` with ``x * log(y)`` where ``x != 0``
        and ``0`` elsewhere, squeezed to remove length-1 dimensions.
    """

    x = np.array(x, ndmin=1)
    y = np.array(y, ndmin=1)

    results = np.zeros_like(y)

    idx = x != 0

    results[idx] = x[idx] * np.log(y[idx])

    return np.squeeze(results)


def size_one_or_n(value, other_array, name):
    """Broadcast a scalar or length-1 value to match another array's length.

    If ``value`` has exactly one element it is broadcast to
    ``other_array.shape[0]``; otherwise its length must already equal
    ``other_array.shape[0]``.

    Args:
        value: Scalar, length-1 sequence, or sequence of the same length
            as ``other_array``. Converted to a 1-D ``float64`` array.
        other_array: Reference array whose first-axis length determines
            the target size.
        name: Name of the parameter, used in the assertion error message.

    Returns:
        A 1-D ``numpy.ndarray`` of ``float64`` with length equal to
        ``other_array.shape[0]``.

    Raises:
        AssertionError: If ``value`` has more than one element and its
            length does not match ``other_array.shape[0]``.
    """

    value_ = np.array(value, dtype=float, ndmin=1)

    if value_.shape[0] == 1:
        value_ = np.zeros(other_array.shape[0], dtype=float) + value

    else:
        assert value_.shape[0] == other_array.shape[0], (
            f'The size of {name} must be either 1 or the same size of n'
        )

    return value_


def pgsig(n, b, sigma):
    r"""Compute Gaussian-background significance for observed counts.

    Returns the significance (in standard deviations) for observing ``n``
    counts when the expected background ``b`` is measured with Gaussian
    uncertainty ``sigma``.  The MLE background estimate
    :math:`\\hat{B}_0` is derived analytically from the profile likelihood
    under the null hypothesis, and the sign follows the direction of the
    excess (positive when ``n >= b``, negative otherwise).

    Args:
        n: Observed counts. May be a scalar or array-like.
        b: Background estimate returned by the background-estimation method.
            Treated as a Gaussian random variable with standard deviation
            ``sigma``. Same shape as ``n`` or broadcastable to it.
        sigma: Standard deviation of the background estimate. May be a
            scalar (applied uniformly) or an array matching ``n``.

    Returns:
        Significance in units of Gaussian standard deviations (z score),
        squeezed to remove length-1 dimensions. Positive values indicate
        an excess above background; negative values indicate a deficit.
    """

    n_ = np.array(n, dtype=float, ndmin=1)
    b_ = np.array(b, dtype=float, ndmin=1)

    sigma_ = size_one_or_n(sigma, n_, 'sigma')

    # Assign sign depending on whether n_ > b_

    sign = np.where(n_ >= b_, 1, -1)

    B0_mle = 0.5 * (
        b_ - sigma_**2 + sqrt(b_**2 - 2 * b_ * sigma_**2 + 4 * n_ * sigma_**2 + sigma_**4)
    )

    # B0_mle could be slightly < 0 (even though it shouldn't) because of the
    # limited numerical precision of the calculator. let's accept as negative as 0.01, and clip
    # at zero to avoid giving results difficult to interpret

    assert np.all(B0_mle > -0.01), 'This is a bug. B0_mle cannot be negative.'

    B0_mle = np.clip(B0_mle, 0, None)

    return squeeze(
        sqrt(2)
        * sqrt(xlogyv(n_, n_ / B0_mle) + (b_ - B0_mle) ** 2 / (2 * sigma_**2) + B0_mle - n_)
        * sign
    )


def _li_and_ma(n_, b_, alpha):

    # In order to avoid numerical problem with 0 * log(0),
    # we add a tiny number to n_ and b_, inconsequential for the computation

    n_ += 1e-25  # type: np.ndarray
    b_ += 1e-25  # type: np.ndarray

    # Pre-compute for speed
    n_plus_b = n_ + b_
    ap1 = alpha + 1

    res = n_ * np.log(ap1 / alpha * (n_ / n_plus_b))

    res += b_ * np.log(ap1 * (b_ / n_plus_b))

    return np.sqrt(2 * res)


def _likelihood_with_sys(o, b, a, s, k, B, M):
    # Keep away from unphysical solutions during maximization
    # (where the argument of the logarithm is negative) by returning
    # a large negative number

    if M + a * B <= 0 or k + 1 <= 0 or B <= 0:
        return -1000

    # Pre-compute for speed

    Ba = B * a
    Bak = B * a * k

    res = -Bak - Ba - B - M + xlogyv(b, B) - k**2 / (2 * s**2) + xlogyv(o, Bak + Ba + M)

    return res


def _get_TS_by_numerical_optimization(n_, b_, alpha, sigma):

    # Numerical optimization to find the maximum likelihood value
    # for the null hypothesis (M=0). We use the closed form for B_mle provided in the paper
    # so that the optimization is in one variable (kk)
    # NOTE: optimize.minimize minimizes, so we multiply the log-likelihood by -1

    def wrapper(kk):
        return -1 * _likelihood_with_sys(
            n_, b_, alpha, sigma, kk, B=(b_ + n_) / (alpha * kk + alpha + 1), M=0
        )

    res = scipy.optimize.minimize(wrapper, [0.0], tol=1e-3)

    # Get the minimum of the -log likelihood
    h0_mlike_value = res['fun']

    # Get alternative hypothesis minimum of the -log likelihood

    h1_mlike_value = -(xlogy(b_, b_) - b_ + xlogy(n_, n_) - n_)

    # Compute Test Statistic of Likelihood Ratio Test

    TS = 2 * (h0_mlike_value - h1_mlike_value)

    return TS


_get_TS_by_numerical_optimization_v = np.vectorize(_get_TS_by_numerical_optimization)


def ppsig(n, b, alpha, sigma=0, k=0):
    r"""Compute Poisson-background significance with optional systematic uncertainty.

    Returns the significance (z score) for detecting ``n`` source counts when
    :math:`\\alpha \\times B` background counts are expected.  The method applied
    depends on the values of ``sigma`` and ``k``:

    - ``sigma=0, k=0`` (default): no systematic error; uses the classic
      Li & Ma (1983) formula.
    - ``k > 0``: uses Eq. 7 from Vianello (2018), treating ``k`` as the
      upper boundary on the fractional systematic uncertainty; ``sigma``
      is ignored.
    - ``sigma > 0``: uses Eq. 9 from Vianello (2018), assuming a Gaussian
      distribution for the systematic uncertainty; ``k`` is ignored.

    Args:
        n: Observed counts. May be a scalar or array-like.
        b: Expected background counts. May be a scalar or array-like of the
            same shape as ``n``.
        alpha: Ratio of source-region to background-region observation
            efficiency. Either a scalar or an array matching ``n``.
        sigma: Standard deviation for the Gaussian systematic case. Either
            a scalar (applied uniformly) or an array matching ``n``.
            Set to ``0`` (default) to disable.
        k: Upper boundary on the fractional systematic uncertainty for the
            bounded systematic case. Either a scalar or an array matching
            ``n``. Set to ``0`` (default) to disable.

    Returns:
        Significance in units of Gaussian standard deviations (z score),
        squeezed to remove length-1 dimensions. Positive values indicate
        an excess; negative values indicate a deficit.

    Note:
        The no-systematic case implements Eq. 17 from Li & Ma (1983),
        ApJ, 272, 317. The systematic cases implement Eqs. 7 and 9 from
        Vianello (2018), ApJS, 236, 17
        (https://doi.org/10.3847/1538-4365/aab780).
    """

    # Make sure we are dealing with arrays, and if not, make the input so. This way
    # we can unify the treatment

    n_ = np.array(n, dtype=float, ndmin=1)
    b_ = np.array(b, dtype=float, ndmin=1)

    k_ = size_one_or_n(k, n_, 'k')

    sigma_ = size_one_or_n(sigma, n_, 'sigma')

    alpha_ = size_one_or_n(alpha, n_, 'alpha')

    # Assign sign depending on whether n_ > b_

    sign = np.where(n_ >= alpha_ * b_, 1, -1)

    # Prepare vector for results
    res = np.zeros(n_.shape[0], dtype=float)

    # Select elements where we need to apply Li & Ma and apply it
    idx_lima = (sigma_ == 0) & (k_ == 0)

    res[idx_lima] = _li_and_ma(n_[idx_lima], b_[idx_lima], alpha_[idx_lima])

    # Select elements where we need to apply eq. 7 from Vianello (2018),
    # which is simply Li & Ma with alpha -> alpha * (k+1)

    idx_eq7 = k_ > 0
    res[idx_eq7] = _li_and_ma(n_[idx_eq7], b_[idx_eq7], alpha_[idx_eq7] * (k_[idx_eq7] + 1))

    # Select elements where we need to apply eq. 9 from Vianello (2018)
    idx_eq9 = sigma_ > 0

    if np.any(idx_eq9):
        TS = _get_TS_by_numerical_optimization_v(
            n_[idx_eq9], b_[idx_eq9], alpha_[idx_eq9], sigma_[idx_eq9]
        )

        res[idx_eq9] = np.sqrt(TS)

    # Return significance

    return np.squeeze(sign * res)
