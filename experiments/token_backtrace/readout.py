"""Token-wise evidence contrasts; no neighbouring-token risk aggregation."""

import numpy as np

from experiments.native_support.unified.calibration import transform


METHODS = ('token_pair', 'odds_pair', 'odds_full')
PRIMARY = 'odds_full'
PROBABILITY_RESOLUTION = np.finfo(np.float32).eps


def log_odds(logp):
    """Regularize the complementary mass at the cached FP32 resolution.

    Cached log probabilities can round to zero. This is a resolution-limited
    contrast, not exact log odds when the complementary mass is unresolved.
    """
    logp = np.minimum(np.asarray(logp, dtype=float), 0.)
    complement = np.maximum(-np.expm1(logp), PROBABILITY_RESOLUTION)
    return logp - np.log(complement)


def contrasts(present_full, absent_full, present_local, absent_local, route):
    full = log_odds(absent_full) - log_odds(present_full)
    local = log_odds(absent_local) - log_odds(present_local)
    return dict(token_pair=.5 * (absent_full - present_full + absent_local - present_local),
                odds_pair=.5 * (full + local), odds_full=full, route=route)


def pack_contrasts(pack, metadata):
    observed = dict(zip(metadata['observations'], pack['observations'].T))
    context = dict(zip(metadata['context'], pack['context'].T))
    present_full = observed['with_full_logp']
    present_local = observed['with_local_logp']
    # Recover the original scalar measurements from the lossless pack layout.
    absent_full = present_full + observed['full_deviation'] + context['full_unit']
    absent_local = present_local + observed['local_deviation'] + context['local_unit']
    return contrasts(present_full, absent_full, present_local, absent_local, observed['route'])


def score_contrasts(values, scales):
    # Preserve the historical 3:1 source/route weight, without its window or
    # unit-mean broadcast. Ranks are reference percentiles, not error chances.
    route_rank = transform(values['route'], scales['route'])
    return {name: .75 * transform(values[name], scales[name]) + .25 * route_rank
            for name in METHODS}
