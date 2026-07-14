from jax import numpy as np
from jax.lax import stop_gradient

from pysages.colvars.orientation import kabsch


def msd(x, ref): #reactive-md
    """Mean-squared displacement of ``x`` from ``ref`` after optimal Kabsch alignment.
    Arguments
    ---------
    x:
        (N, 3) current atomic positions of the aligned group.
    ref:
        (N, 3) reference atomic positions.
    """
    P = x - x.mean(axis=0)               # center current structure
    Q = ref - ref.mean(axis=0)           # center reference
    U = stop_gradient(kabsch(P, Q))     
    d = P @ U - Q                        # aligned current minus reference (kabsch aligns as P @ U)
    return np.sum(d * d) / P.shape[0]
