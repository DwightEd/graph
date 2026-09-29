"""Bounded observation corrections, regularized by measured message overlap."""
import numpy as np
from scipy.optimize import minimize

BOUND = .1
PENALTY = .05


def divergence(dual, count):
    result = np.zeros(count)
    result[:-1] -= dual
    result[1:] += dual
    return result


def solve_correction(innovation, edges, bound=BOUND, penalty=PENALTY):
    """Solve box-constrained weighted TV through its exact convex dual."""
    innovation = np.asarray(innovation,dtype=float)
    limits = penalty*np.asarray(edges,dtype=float)
    if len(innovation)==1 or not np.any(limits):
        return np.clip(innovation,-bound,bound),0.

    def objective(dual):
        correction = np.clip(innovation-divergence(dual,len(innovation)),-bound,bound)
        differences = np.diff(correction)
        dual_value = .5*np.sum((correction-innovation)**2)+dual@differences
        return -dual_value,-differences

    optimum = minimize(objective,np.zeros(len(edges)),jac=True,method='L-BFGS-B',
        bounds=list(zip(-limits,limits)),options=dict(ftol=1e-14,gtol=1e-10,maxiter=5000,maxls=50))
    correction = np.clip(innovation-divergence(optimum.x,len(innovation)),-bound,bound)
    differences = np.diff(correction)
    gap = float(limits@np.abs(differences)-optimum.x@differences)
    if gap>1e-6:
        raise RuntimeError(f'TV dual gap {gap:g}; solver: {optimum.message}')
    return correction,gap


def overlap_parts(attention, effect):
    """All heads, queries, absolute keys; preserve the two effect signs."""
    energy = np.abs(effect).sum(-1,dtype=np.float64)
    reading = attention.astype(np.float64)
    reading /= np.maximum(reading.sum(-1,keepdims=True),1e-30)
    root_reading = np.sqrt(reading)
    read_overlap = (root_reading[:,1:]*root_reading[:,:-1]).sum(-1)
    signed_overlap = np.zeros_like(read_overlap)
    for sign in (1.,-1.):
        root = np.sqrt(np.maximum(sign*effect,0).astype(np.float64))
        signed_overlap += (root[:,1:]*root[:,:-1]).sum(-1)
    return energy,read_overlap*signed_overlap


def combine_overlap(energy, overlap):
    total = energy.sum(0)
    denominator = np.sqrt(total[1:]*total[:-1])
    return np.divide(overlap.sum(0),denominator,out=np.zeros_like(denominator),where=denominator>0)


def score_observations(base, local, edges, shuffled):
    innovation = local-base
    values = dict(base=base,token_observation=local,
        clipped=base+np.clip(innovation,-BOUND,BOUND))
    gaps = {}
    for name,weights in (('anchored_flow',edges),('uniform_tv',np.ones_like(edges)),
                         ('shuffled_tv',shuffled)):
        correction,gaps[name] = solve_correction(innovation,weights)
        values[name] = base+correction
    return values,gaps
