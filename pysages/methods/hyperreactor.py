# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
Hyperreactor -- ab initio Hyperreactor Dynamics (Stan-Bernhardt/Ochsenfeld 2024).  #reactive-md

HRD combines two biases that complement each other:
  * a hyperdynamics BOOST (aMD/GaMD/SaMD) that flattens low-energy basins (as in gamd.py), and
  * a periodic spherical CONFINEMENT piston that squeezes the cluster (as in confinement.py).

The total added bias is the SUM of the two, kept as two separate additive terms:
    bias = f'(V) * F_phys        (boost; added on top of the physical force)
         + (-grad V_sphere)      (confinement; only after equilibration)

Lifecycle (keyed on the guarded real-step count), following the HRD protocol:
  * nsteps <  init_steps  : collect energy statistics; no boost, no sphere.
  * init <= nsteps < equil: boost on, E/k0 updating; sphere still OFF.
  * nsteps >= equil_steps : boost on with E/k0 FROZEN; sphere turned ON (piston starts).

The boost machinery (welford_var, mode parameters, dV/f') is imported from gamd.py; the sphere
force is built by confinement.get_sphere_force(...) and passed in as `ext_force`. This method
needs the energy/force patch (reads data.energy and data.forces).
"""
from functools import partial

import jax.numpy as np

from pysages.methods.core import SamplingMethod, generalize
from pysages.methods.gamd import _MODES, welford_var
from pysages.methods.utils import position_guard
from pysages.typing import JaxArray, NamedTuple


class HyperreactorState(NamedTuple):
    """
    xi:             collective variable value (logged).
    bias:           total bias force (boost + sphere) added to the physical force.
    prev_positions: positions at the previous update, for the position guard.
    count, mean, m2, pot_min, pot_max: Welford energy statistics for the boost.
    E, k0:          boost reference energy / force constant (frozen after equil_steps).
    proj:           logged diagnostics [dV (boost potential), # atoms outside the sphere wall].
    ncalls:         raw update calls (~2x real steps on ASE; diagnostics).
    nsteps:         guarded real MD-step count -> boost lifecycle + sphere gate/clock.
    """

    xi: JaxArray
    bias: JaxArray
    prev_positions: JaxArray
    count: float
    mean: float
    m2: float
    pot_min: float
    pot_max: float
    E: float
    k0: float
    proj: JaxArray
    ncalls: int = 0
    nsteps: int = 0

    def __repr__(self):
        return repr("PySAGES" + type(self).__name__)


class Hyperreactor(SamplingMethod):
    """Ab initio Hyperreactor Dynamics: hyperdynamics boost + gated sphere piston.

    Arguments
    ---------
    cvs:
        Collective variable(s), kept for logging only (`cv_grad=False`).
    parameter:
        sigma0 (eV) for gamd_*/samd, or alpha for amd (the boost strength).
    init_steps:
        steps of pure statistics collection before the boost turns on.
    equil_steps:
        step at which the boost freezes AND the sphere piston turns on.

    Keyword arguments
    -----------------
    ext_force:
        the sphere force callable from `confinement.get_sphere_force(...)`.
    mode:
        boost mode: "gamd_lower" (default), "gamd_upper", "amd", "samd".
    c0:
        samd floor constant (default 1e-4); ignored by other modes.
    """

    snapshot_flags = {"positions", "indices", "masses", "forces", "energy"}

    def __init__(self, cvs, parameter, init_steps, equil_steps,
                 mode="gamd_lower", c0=1e-4, **kwargs):
        kwargs["cv_grad"] = False
        super().__init__(cvs, **kwargs)
        if mode not in _MODES:
            raise ValueError(f"unknown boost mode {mode!r}; choose from {list(_MODES)}")
        self.parameter = parameter
        self.init_steps = init_steps
        self.equil_steps = equil_steps
        self.mode = mode
        self.c0 = c0

    def build(self, snapshot, helpers, *args, **kwargs):
        self.ext_force = self.kwargs.get("ext_force", None)
        return _hyperreactor(self, snapshot, helpers)


def _hyperreactor(method, snapshot, helpers):
    cv = method.cv
    dt = snapshot.dt
    natoms = np.size(snapshot.positions, 0)
    dim = helpers.dimensionality()
    init_steps = method.init_steps
    equil_steps = method.equil_steps
    param = method.parameter
    ext_force = method.ext_force                       # sphere force from get_sphere_force(...)

    _params, _fprime, _dV = _MODES[method.mode]
    if method.mode == "samd":
        _params = partial(_params, c0=method.c0)
        _fprime = partial(_fprime, c0=method.c0)
        _dV = partial(_dV, c0=method.c0)

    def initialize():
        xi = cv(helpers.query(snapshot))
        bias = np.zeros((natoms, dim))
        return HyperreactorState(
            xi, bias, snapshot.positions,
            0.0, 0.0, 0.0, np.inf, -np.inf,            # count, mean, m2, pot_min, pot_max
            0.0, 0.0,                                  # E, k0
            np.zeros(2),                               # proj = [dV, #outside]
            0, 0,
        )

    def update(state, data):
        V = data.energy
        Fphys = data.forces
        is_new = position_guard(state.prev_positions, data.positions)
        ncalls = state.ncalls + 1
        nsteps = state.nsteps + np.where(is_new, 1, 0)

        # ---- boost lifecycle (identical to gamd.py) ----
        do_stat = is_new & (nsteps < equil_steps)
        count_t = state.count + 1.0
        mean_t, m2_t, _ = welford_var(count_t, state.mean, state.m2, V)
        count = np.where(do_stat, count_t, state.count)
        mean = np.where(do_stat, mean_t, state.mean)
        m2 = np.where(do_stat, m2_t, state.m2)
        pot_min = np.where(do_stat, np.minimum(state.pot_min, V), state.pot_min)
        pot_max = np.where(do_stat, np.maximum(state.pot_max, V), state.pot_max)

        var = np.where(count > 2, m2 / count, 0.0)
        vstd = np.sqrt(np.maximum(var, 1e-12))
        E_new, k0_new = _params(pot_min, pot_max, mean, vstd, param)
        recompute = (nsteps >= init_steps) & (nsteps < equil_steps)
        E = np.where(recompute, E_new, state.E)
        k0 = np.where(recompute, k0_new, state.k0)

        boosting = nsteps >= init_steps
        coeff = np.where(boosting, _fprime(V, E, k0, pot_min, pot_max), 0.0)
        boost_bias = coeff * Fphys                     # f'(V) * F_phys
        dV = np.where(boosting, _dV(V, E, k0, pot_min, pot_max), 0.0)

        # ---- sphere piston, gated ON only after equilibration ----
        sphere_on = nsteps >= equil_steps
        t_sphere = np.maximum(0.0, (nsteps - equil_steps).astype(dt.dtype)) * dt   # clock starts at equil
        sforce, sproj = ext_force(data, t_sphere)      # +grad(U_sphere), # atoms outside wall
        sphere_bias = np.where(sphere_on, -sforce, 0.0)

        # ---- total bias = boost + sphere (two separate additive terms) ----
        bias = boost_bias + sphere_bias
        proj = np.array([dV, np.where(sphere_on, sproj[0], 0.0)])
        xi = cv(data)
        return HyperreactorState(xi, bias, data.positions, count, mean, m2, pot_min, pot_max,
                                 E, k0, proj, ncalls, nsteps)

    return snapshot, initialize, generalize(update, helpers)
