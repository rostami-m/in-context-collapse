"""Exact evaluation of paired integer bootstrap estimates.

Random streams, image units and linear/type-7 percentile definitions are
unchanged. Divide only after signed integer sums and rational interpolation;
strict scientific boundaries must use the retained rationals, not display floats.
"""
from fractions import Fraction
import random


def type7_integer_quantile(ordered, probability, denominator):
    probability = Fraction(probability)
    if not ordered or not 0 <= probability <= 1 or type(denominator) is not int or denominator <= 0:
        raise ValueError('invalid exact quantile inputs')
    position = probability * (len(ordered) - 1)
    lower = position.numerator // position.denominator
    upper = min(lower + 1, len(ordered) - 1)
    value = int(ordered[lower]) + (position - lower) * (int(ordered[upper]) - int(ordered[lower]))
    return value / denominator


def exact_estimate(value, field):
    row = value['exact_rationals'][field]
    return Fraction(row['numerator'], row['denominator'])


def integer_image_interval(numerators, denominator, *, confidence=.95, draws=10000,
                           seed=90210, generator='python'):
    """Bootstrap per-image sums on a known integer lattice without rounding."""
    numerators = list(numerators)
    if (not numerators or any(type(v) is not int for v in numerators)
            or type(denominator) is not int or denominator <= 0
            or type(draws) is not int or draws < 2):
        raise ValueError('integer images, positive denominator and >=2 draws required')
    probability = Fraction(str(confidence))
    if not 0 < probability < 1:
        raise ValueError('confidence must be strictly between zero and one')
    n = len(numerators)
    if generator == 'python':
        rng = random.Random(seed)
        totals = sorted(sum(rng.choices(numerators, k=n)) for _ in range(draws))
    elif generator == 'numpy':
        import numpy as np
        # NumPy's signed integer summation is exact only within int64 bounds.
        if max(map(abs, numerators)) * n > np.iinfo(np.int64).max:
            raise ValueError('integer bootstrap exceeds exact int64 range')
        values = np.asarray(numerators, dtype=np.int64)
        rng = np.random.default_rng(seed)
        totals = []
        for offset in range(0, draws, 500):
            indices = rng.integers(0, n, (min(500, draws-offset), n))
            totals.extend(map(int, values[indices].sum(axis=1)))
        totals.sort()
    else:
        raise ValueError('unknown bootstrap generator')
    tail = (1 - probability) / 2
    rational = {'mean': Fraction(sum(numerators), denominator*n),
                'low': type7_integer_quantile(totals, tail, denominator*n),
                'high': type7_integer_quantile(totals, 1-tail, denominator*n)}
    return {**{key: float(value) for key, value in rational.items()},
            'exact_rationals': {key: {'numerator': value.numerator, 'denominator': value.denominator}
                                for key, value in rational.items()},
            'confidence': float(probability), 'n_images': n, 'draws': draws, 'seed': seed,
            'rng': 'random.Random' if generator=='python' else 'numpy.random.default_rng',
            'quantile': 'linear/type-7; exact rational interpolation of integer sums'}
