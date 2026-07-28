# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
Structural (RMSD) metadynamics for reactive exploration -- Grimme's CREST MTD(RMSD).  #reactive-md

A repulsive Gaussian "hill" is deposited on the structural distance to each visited reference
structure, pushing the system into new geometries. Because RMSD^2 == MSD, the CREST hill
exp(-alpha * RMSD^2) is exactly exp(-alpha * MSD), so we build the bias on the smooth `msd`
helper (below) and never hit the 1/RMSD singularity.

Bias potential over a growing set of reference structures {ref_i}:
    V(x) = sum_i  k * f_dmp_i * exp(-alpha * MSD(x, ref_i))
with a switch-on ramp f_dmp_i so a freshly deposited hill grows in smoothly (no force jump).
A new reference is deposited every `stride` real MD steps, up to `nmax`. The additive bias force
is -grad(V), from jax.grad (differentiating MSD, Kabsch held fixed by stop_gradient).

Optionally, a STATIC confining wall (from confinement.get_logfermi_wall) can be supplied via
`wall=`; its force is added to the bias to keep gas-phase molecules together during exploration,
exactly as CREST does. No energy patch is required.
"""
from functools import partial

import jax
from jax import grad
from jax import numpy as np
from jax.lax import stop_gradient

from pysages.colvars.orientation import kabsch
from pysages.methods.core import SamplingMethod, generalize
from pysages.methods.utils import position_guard
from pysages.typing import JaxArray, NamedTuple


def msd(x, ref):
    """Mean-squared displacement of ``x`` from ``ref`` after optimal Kabsch alignment.

    Both are centered internally, so the result is invariant to rigid translation/rotation.
    The Kabsch rotation is wrapped in stop_gradient (envelope theorem -> exact gradient), and
    we differentiate MSD rather than RMSD to avoid the 1/RMSD singularity at the reference.
    """
    P = x - x.mean(axis=0)
    Q = ref - ref.mean(axis=0)
    U = stop_gradient(kabsch(P, Q))
    d = P @ U - Q
    return np.sum(d * d) / P.shape[0]


# ============================ bias potential ============================
def _bias_potential(x, references, dep_steps, active, nsteps, k, alpha, kappa):
    """Sum of switch-on-damped Gaussian hills on MSD to each active reference. Returns a scalar."""
    def one(ref, dep, act):
        m = msd(x, ref)
        fdmp = 2.0 / (1.0 + np.exp(-kappa * (nsteps - dep))) - 1.0   # 0 at deposition -> 1 later
        fdmp = np.clip(fdmp, 0.0, 1.0)
        return act * k * fdmp * np.exp(-alpha * m)
    return np.sum(jax.vmap(one)(references, dep_steps, active))


# ============================ state + method ============================
class RMSDMetadState(NamedTuple):
    """
    xi:             smallest MSD to any deposited reference (logged; 0-d).
    bias:           the metadynamics (+ optional wall) bias force added to the physical force.
    prev_positions: positions at the previous update, for the position guard.
    references:     (nmax, natoms, 3) deposited reference structures (unused slots are zero).
    dep_steps:      (nmax,) real-step at which each reference was deposited (for the damping ramp).
    n_refs:         number of references deposited so far.
    proj:           logged diagnostic (# hills deposited).
    ncalls:         raw update calls (~2x real steps on ASE; diagnostics).
    nsteps:         guarded real MD-step count -> deposition clock + damping.
    """

    xi: JaxArray
    bias: JaxArray
    prev_positions: JaxArray
    references: JaxArray
    dep_steps: JaxArray
    n_refs: int
    proj: JaxArray
    ncalls: int = 0
    nsteps: int = 0

    def __repr__(self):
        return repr("PySAGES" + type(self).__name__)


class RMSDMetadynamics(SamplingMethod):
    """Grimme CREST-style metadynamics on structural (RMSD/MSD) distance.

    Keyword arguments
    -----------------
    height:  Gaussian hill height k (eV). Default 0.5.
    alpha:   Gaussian width parameter (1/A^2); larger = narrower hills. Default 1.0.
    stride:  deposit a new reference every this many real MD steps. Default 100.
    nmax:    maximum number of reference structures. Default 50.
    kappa:   switch-on damping rate per step (CREST uses 0.03). Default 0.03.
    wall:    optional static confinement force from confinement.get_logfermi_wall(...);
             its force is added to the bias to keep molecules together. Default None.
    """

    snapshot_flags = {"positions", "indices"}

    def __init__(self, cvs, height=0.5, alpha=1.0, stride=100, nmax=50, kappa=0.03, **kwargs):
        kwargs["cv_grad"] = False
        super().__init__(cvs, **kwargs)
        self.height = height
        self.alpha = alpha
        self.stride = stride
        self.nmax = nmax
        self.kappa = kappa

    def build(self, snapshot, helpers, *args, **kwargs):
        self.wall = self.kwargs.get("wall", None)
        return _rmsd_metad(self, snapshot, helpers)


def _rmsd_metad(method, snapshot, helpers):
    cv = method.cv
    natoms = np.size(snapshot.positions, 0)
    dim = helpers.dimensionality()
    k = method.height
    alpha = method.alpha
    stride = method.stride
    nmax = method.nmax
    kappa = method.kappa
    wall = method.wall                                   # optional static confining wall

    def initialize():
        bias = np.zeros((natoms, dim))
        references = np.zeros((nmax, natoms, dim))
        dep_steps = np.zeros(nmax)
        xi = np.asarray(0.0)
        return RMSDMetadState(xi, bias, snapshot.positions, references, dep_steps,
                              0, np.zeros(1), 0, 0)

    def update(state, data):
        x = data.positions[:, :3]
        is_new = position_guard(state.prev_positions, data.positions)
        ncalls = state.ncalls + 1
        nsteps = state.nsteps + np.where(is_new, 1, 0)

        # --- deposit a new reference once per `stride` real steps, while there is room ---
        do_deposit = is_new & (nsteps % stride == 0) & (state.n_refs < nmax) & (nsteps > 0)
        idx = state.n_refs
        references = jax.lax.cond(
            do_deposit, lambda r: r.at[idx].set(x), lambda r: r, state.references)
        dep_steps = jax.lax.cond(
            do_deposit, lambda d: d.at[idx].set(nsteps.astype(d.dtype)), lambda d: d,
            state.dep_steps)
        n_refs = state.n_refs + np.where(do_deposit, 1, 0)

        # --- bias force = -grad(V) from the Gaussian hills ---
        active = (np.arange(nmax) < n_refs).astype(references.dtype)
        t = nsteps.astype(references.dtype)
        gV = grad(_bias_potential)(x, references, dep_steps, active, t, k, alpha, kappa)
        bias = -gV.reshape(state.bias.shape)

        # --- optional static confining wall (added, kept a separate term) ---
        if wall is not None:
            wforce, _ = wall(data, 0.0)
            bias = bias - wforce.reshape(state.bias.shape)

        # --- diagnostics ---
        msds = jax.vmap(lambda ref: msd(x, ref))(references)
        xi = np.min(np.where(active > 0, msds, np.inf))
        xi = np.where(n_refs > 0, xi, np.asarray(0.0))
        proj = np.asarray(n_refs, dtype=references.dtype).reshape(1)

        return RMSDMetadState(xi, bias, data.positions, references, dep_steps,
                              n_refs, proj, ncalls, nsteps)

    return snapshot, initialize, generalize(update, helpers)
