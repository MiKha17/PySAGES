# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
Confinement -- the Wang/Martinez nanoreactor sphere piston (+ Stan 2022 smooth schedules).  #reactive-md

A time-dependent spherical wall squeezes a gas-phase cluster (contract) and lets it relax (expand),
driving reactions. The method is a thin time-dependent-restraint driver (reusing Gustavo
Perez-Lemus's Nanoreactor architecture) plus the ASE position guard (Step 2), so the piston clock
advances once per real MD step.

Physics is injected as an `ext_force` callable built by `get_sphere_force(...)`, which supports the
three Stan-2022 radius schedules (all with the mass-weighted half-harmonic wall):
  * "step"        -- Wang/Martinez rectangular wave: two fixed walls at r_max/r_min, hard-switched.
  * "cosine"      -- one wall gliding sinusoidally between r_min and r_max.
  * "smooth_step" -- one wall following sin(pi/2 * cos(2 pi t/T)) capped at r_max (spends longer
                     expanded than contracted; the HRD default).
The factory returns +grad(U) with the atomic-mass factor applied ONCE (equal acceleration); the
driver applies bias = -force.

`get_logfermi_wall(...)` provides a separate, STATIC soft confinement (xtb log-Fermi form,
V = sum_A kB*T*ln(1 + exp(beta*(r_A - R0)))), NOT mass-weighted -- used to keep gas-phase molecules
together during RMSD metadynamics (Step 6). It has the same (data, t) -> (force, proj) signature.
"""
from functools import partial

import jax.numpy as np
from jax import grad, jit

from pysages.methods.core import SamplingMethod, generalize
from pysages.methods.utils import position_guard
from pysages.typing import JaxArray, NamedTuple

_KB_EV = 8.617333262e-5   # Boltzmann constant in eV/K


# ============================ state + driver ============================
class ConfinementState(NamedTuple):
    """
    xi:             collective variable value (logged only).
    bias:           the confinement bias force added to the physical force.
    prev_positions: positions at the previous update, for the position guard.
    proj:           logged diagnostic (# atoms outside the active wall).
    ncalls:         raw update calls (~2x real steps on ASE; diagnostics).
    nsteps:         guarded real MD-step count -> drives the piston clock.
    """

    xi: JaxArray
    bias: JaxArray
    prev_positions: JaxArray
    proj: JaxArray
    ncalls: int = 0
    nsteps: int = 0

    def __repr__(self):
        return repr("PySAGES" + type(self).__name__)


class Confinement(SamplingMethod):
    """Time-dependent spherical confinement (Wang/Martinez nanoreactor piston).

    Keyword arguments
    -----------------
    ext_force:
        A callable `(data, t) -> (force, proj)` built by `get_sphere_force(...)`.
    """

    snapshot_flags = {"positions", "indices", "masses"}

    def __init__(self, cvs, **kwargs):
        kwargs["cv_grad"] = False
        super().__init__(cvs, **kwargs)

    def build(self, snapshot, helpers, *args, **kwargs):
        self.ext_force = self.kwargs.get("ext_force", None)
        return _confinement(self, snapshot, helpers)


def _confinement(method, snapshot, helpers):
    cv = method.cv
    dt = snapshot.dt
    natoms = np.size(snapshot.positions, 0)
    ext_force = method.ext_force

    def initialize():
        xi = cv(helpers.query(snapshot))
        bias = np.zeros((natoms, helpers.dimensionality()))
        return ConfinementState(xi, bias, snapshot.positions, np.zeros(1), 0, 0)

    def update(state, data):
        is_new = position_guard(state.prev_positions, data.positions)
        ncalls = state.ncalls + 1
        nsteps = state.nsteps + np.where(is_new, 1, 0)      # guarded real-step count
        xi = cv(data)
        force, proj = ext_force(data, nsteps * dt)          # piston clock on guarded steps
        bias = -force.reshape(state.bias.shape)
        return ConfinementState(xi, bias, data.positions, proj, ncalls, nsteps)

    return snapshot, initialize, generalize(update, helpers)


# ============================ sphere piston physics ============================
class _SphereParams(NamedTuple):
    k_max: float
    k_min: float
    r_max: float
    r_min: float
    period: float
    expand_frac: float
    mass_weight: bool


def _switch(t, P):
    """Martinez rectangular wave f(t): 1.0 during the expand phase, 0.0 during contract."""
    frac = t / P.period - np.floor(t / P.period)
    return np.where(frac < P.expand_frac, 1.0, 0.0)


def _radius_cosine(t, P):
    """Cosine schedule: one wall gliding between r_min and r_max."""
    theta = 2.0 * np.pi * t / P.period
    return 0.5 * (P.r_max + P.r_min) + 0.5 * (P.r_max - P.r_min) * np.cos(theta)


def _radius_smooth_step(t, P):
    """Stan 2022 smooth-step schedule: sin(pi/2 * cos(...)) capped at r_max (longer expanded)."""
    theta = 2.0 * np.pi * t / P.period
    return np.minimum(P.r_max, P.r_max + (P.r_max - P.r_min) * np.sin(0.5 * np.pi * np.cos(theta)))


def _sphere_potential_step(pos, t, P):
    """Rectangular two-wall potential (Wang/Martinez), MASS-FREE."""
    r = np.linalg.norm(pos, axis=1)
    f = _switch(t, P)

    def outer(r0, k):
        F = r - r0
        return 0.5 * k * np.where(F < 0.0, 0.0, F * F)

    return np.sum(f * outer(P.r_max, P.k_max) + (1.0 - f) * outer(P.r_min, P.k_min))


def _sphere_potential_moving(pos, t, P, radius_fn):
    """Single half-harmonic wall at a smoothly moving radius r0(t), MASS-FREE."""
    r = np.linalg.norm(pos, axis=1)
    F = r - radius_fn(t, P)
    return np.sum(0.5 * P.k_max * np.where(F < 0.0, 0.0, F * F))


def _external_sphere(data, t, P, potential, active_radius):
    pos = data.positions[:, :3]
    g = grad(potential, argnums=0)(pos, t, P)               # +grad(U), mass-free
    if P.mass_weight:                                       # static bool: trace-time branch
        force = data.masses.flatten()[:, None] * g          # mass factor applied ONCE
    else:
        force = g
    R = active_radius(t, P)
    proj = np.sum(np.where(np.linalg.norm(pos, axis=1) > R, 1.0, 0.0)).reshape(1)
    return force, proj


def get_sphere_force(r_min=8.0, r_max=14.0, k_min=0.5, k_max=1.0,
                     period=1.0, expand_frac=0.75, mass_weight=True, schedule="step"):
    """Factory for the sphere-piston force (fixed origin).

    Returns a jitted `ext_force(data, t) -> (force, proj)` for `Confinement(ext_force=...)`.
    `schedule` is "step" (Wang/Martinez rectangular, default), "cosine", or "smooth_step"
    (Stan 2022). Units: radii in A, force constants in eV/A^2, `period` in ASE time units.
    """
    P = _SphereParams(k_max, k_min, r_max, r_min, period, expand_frac, bool(mass_weight))
    if schedule == "step":
        potential = _sphere_potential_step
        active_radius = lambda t, PP: np.where(_switch(t, PP) > 0.5, PP.r_max, PP.r_min)
    elif schedule == "cosine":
        potential = partial(_sphere_potential_moving, radius_fn=_radius_cosine)
        active_radius = _radius_cosine
    elif schedule == "smooth_step":
        potential = partial(_sphere_potential_moving, radius_fn=_radius_smooth_step)
        active_radius = _radius_smooth_step
    else:
        raise ValueError(f"unknown schedule {schedule!r}; choose step, cosine, or smooth_step")
    return jit(partial(_external_sphere, P=P, potential=potential, active_radius=active_radius))


# ============================ log-Fermi confinement (static wall) ============================
class _WallParams(NamedTuple):
    R0: float
    beta: float
    kT: float          # kB * T_wall (eV)


def _logfermi_potential(pos, P):
    """xtb log-Fermi soft wall: V = sum_A kB*T*ln(1 + exp(beta*(|r_A| - R0))). NOT mass-weighted."""
    r = np.linalg.norm(pos, axis=1)
    return np.sum(P.kT * np.log(1.0 + np.exp(P.beta * (r - P.R0))))


def _external_logfermi(data, t, P):        # t ignored (static wall)
    pos = data.positions[:, :3]
    g = grad(_logfermi_potential)(pos, P)                   # +grad(V); no mass factor
    proj = np.sum(np.where(np.linalg.norm(pos, axis=1) > P.R0, 1.0, 0.0)).reshape(1)
    return g, proj


def get_logfermi_wall(R0=10.0, beta=6.0, temperature=300.0):
    """Factory for a STATIC log-Fermi confinement wall (xtb form).

    Returns a jitted `(data, t) -> (force, proj)` with the same signature as get_sphere_force,
    so it can be added to RMSDMetadynamics (Step 6) or used standalone. `R0` in A, `beta` in 1/A,
    `temperature` in K (sets the prefactor kB*T in eV). Not mass-weighted. The consumer applies
    bias = -force.
    """
    P = _WallParams(R0, beta, _KB_EV * temperature)
    return jit(partial(_external_logfermi, P=P))
