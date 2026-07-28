# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
GaMD -- Gaussian accelerated molecular dynamics boost (with aMD / SaMD modes).  #reactive-md

Instead of a spatial wall, GaMD adds a small upward "boost" to the potential energy in low-energy
regions, flattening basins so the system escapes them faster while leaving high-energy transition
regions (near or above a reference energy E) untouched. The boost force added to the physical force
is f'(V) * F_phys, where V is the physical potential energy, F_phys the physical force, and f'(V)
the derivative of the boost potential dV(V). This requires the energy/force patch (Step 1): the
method reads data.energy and data.forces.

Four modes, selected at construction and bound once at build (no traced string branch):
  * "gamd_lower" / "gamd_upper" -- harmonic boost, dV = 1/2 * k0/(Vmax-Vmin) * (E-V)^2, hard-gated V<E
  * "amd"                       -- Hamelberg boost, dV = (E-V)^2 / (a + (E-V)), hard-gated V<E
  * "samd"                      -- sigmoidal boost (self-limiting, ungated)
All transcribed from ochsenfeld-lab/adaptive_sampling sampling_tools/amd.py and verified numerically
(f'(V) == d/dV[dV](V) for every mode; total force F_phys + f'*F_phys == -grad(V+dV)).

Three-phase lifecycle, keyed on the guarded real-step count:
  * nsteps <  init_steps  : collect energy statistics only; no boost.
  * nsteps == init_steps  : compute E and k0 from the statistics; boost turns on.
  * init <= nsteps < equil: boost on AND E, k0 keep updating with the growing statistics.
  * nsteps >= equil_steps : E, k0 frozen (production); dV is logged for reweighting.

The energy statistics use Welford's online mean/variance, guarded so they advance once per real MD
step (the ASE backend fires update ~2x/step).
"""
from functools import partial

import jax.numpy as np

from pysages.methods.core import SamplingMethod, generalize
from pysages.methods.utils import position_guard
from pysages.typing import JaxArray, NamedTuple


# ============================ online statistics ============================
def welford_var(count, mean, M2, x):
    """Welford online update (matches adaptive_sampling utils.welford_var).

    `count` is the sample number INCLUDING x (caller pre-increments). Returns (mean, M2, var)
    with the population variance M2/count, defined as 0 until count > 2.
    """
    delta = x - mean
    mean = mean + delta / count
    delta2 = x - mean
    M2 = M2 + delta * delta2
    var = np.where(count > 2, M2 / count, 0.0)
    return mean, M2, var


# ============================ mode parameters (E, k0) ============================
# Each returns (E, k0) from the energy statistics. For "amd" the k0 slot carries alpha; for
# "samd" the E slot is unused and k0 carries the force constant k.
def _params_gamd_lower(vmin, vmax, vavg, vstd, param):
    ko = (param / vstd) * ((vmax - vmin) / (vmax - vavg))
    return vmax, np.minimum(1.0, ko)


def _params_gamd_upper(vmin, vmax, vavg, vstd, param):
    ko = (1.0 - param / vstd) * ((vmax - vmin) / (vavg - vmin))
    k0 = np.where((ko > 0.0) & (ko <= 1.0), ko, 1.0)
    return vmin + (vmax - vmin) / k0, k0


def _params_amd(vmin, vmax, vavg, vstd, param):
    return vmax, param                         # E = Vmax; k0 slot carries alpha


def _params_samd(vmin, vmax, vavg, vstd, param, c0):
    ko = (param / vstd) * ((vmax - vmin) / (vmax - vavg))
    k0 = np.minimum(1.0, ko)
    c = 1.0 / c0 - 1.0
    k1 = np.maximum(0.0, (np.log(c) + np.log(vstd / param - 1.0)) / (vavg - vmin))
    k = np.where(vstd / param <= 1.0, k0, np.maximum(k0, k1))
    return vmin, k                             # E slot unused; k0 slot carries k


# ============================ boost potential dV and f' = dV/dV ============================
def _fprime_gamd(V, E, k0, vmin, vmax):
    prefac = k0 / (vmax - vmin)
    return np.where(V < E, -prefac * (E - V), 0.0)


def _dV_gamd(V, E, k0, vmin, vmax):
    prefac = k0 / (vmax - vmin)
    return np.where(V < E, 0.5 * prefac * (E - V) ** 2, 0.0)


def _fprime_amd(V, E, k0, vmin, vmax):
    a = k0
    val = -((V - E) * (V - 2.0 * a - E) / (V - a - E) ** 2)
    return np.where(V < E, val, 0.0)


def _dV_amd(V, E, k0, vmin, vmax):
    a = k0
    return np.where(V < E, (E - V) ** 2 / (a + (E - V)), 0.0)


def _fprime_samd(V, E, k0, vmin, vmax, c0):
    k = k0
    c = 1.0 / c0 - 1.0
    return 1.0 / (np.exp(-k * (V - vmin) + np.log(c)) + 1.0) - 1.0


def _dV_samd(V, E, k0, vmin, vmax, c0):
    k = k0
    c = 1.0 / c0 - 1.0
    return vmax - V - (1.0 / k) * np.log(
        (c + np.exp(k * (vmax - vmin))) / (c + np.exp(k * (V - vmin))))


_MODES = {
    "gamd_lower": (_params_gamd_lower, _fprime_gamd, _dV_gamd),
    "gamd_upper": (_params_gamd_upper, _fprime_gamd, _dV_gamd),
    "amd":        (_params_amd,        _fprime_amd,  _dV_amd),
    "samd":       (_params_samd,       _fprime_samd, _dV_samd),
}


# ============================ state + method ============================
class GaMDState(NamedTuple):
    """
    xi:             collective variable value (logged; used to bin dV for reweighting).
    bias:           the boost force f'(V) * F_phys added to the physical force.
    prev_positions: positions at the previous update, for the position guard.
    count:          Welford sample count of the physical energy.
    mean:           Welford running mean of the physical energy (= pot_avg).
    m2:             Welford running M2 (sum of squared deviations).
    pot_min:        minimum physical energy seen during collection.
    pot_max:        maximum physical energy seen during collection.
    E:              boost reference energy (frozen after equil_steps).
    k0:             boost force constant / alpha (amd) / k (samd); frozen after equil_steps.
    proj:           logged boost potential dV(V) (for reweighting).
    ncalls:         raw update calls (~2x real steps on ASE; diagnostics).
    nsteps:         guarded real MD-step count -> the phase clock.
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


class GaMD(SamplingMethod):
    """Gaussian accelerated MD boost (modes: gamd_lower, gamd_upper, amd, samd).

    Arguments
    ---------
    cvs:
        Collective variable(s), kept for logging / reweighting binning (`cv_grad=False`).
    parameter:
        sigma0 (the target boost std, in eV) for gamd_*/samd, or alpha for amd.
    init_steps:
        steps of pure statistics collection before the boost turns on.
    equil_steps:
        step at which E and k0 freeze (production begins).

    Keyword arguments
    -----------------
    mode:
        one of "gamd_lower" (default), "gamd_upper", "amd", "samd".
    c0:
        samd floor constant (default 1e-4); ignored by the other modes.
    """

    snapshot_flags = {"positions", "indices", "masses", "forces", "energy"}

    def __init__(self, cvs, parameter, init_steps, equil_steps,
                 mode="gamd_lower", c0=1e-4, **kwargs):
        kwargs["cv_grad"] = False
        super().__init__(cvs, **kwargs)
        if mode not in _MODES:
            raise ValueError(f"unknown GaMD mode {mode!r}; choose from {list(_MODES)}")
        self.parameter = parameter
        self.init_steps = init_steps
        self.equil_steps = equil_steps
        self.mode = mode
        self.c0 = c0

    def build(self, snapshot, helpers, *args, **kwargs):
        return _gamd(self, snapshot, helpers)


def _gamd(method, snapshot, helpers):
    cv = method.cv
    natoms = np.size(snapshot.positions, 0)
    dim = helpers.dimensionality()
    init_steps = method.init_steps
    equil_steps = method.equil_steps
    param = method.parameter

    _params, _fprime, _dV = _MODES[method.mode]
    if method.mode == "samd":                  # bind the samd floor constant once
        _params = partial(_params, c0=method.c0)
        _fprime = partial(_fprime, c0=method.c0)
        _dV = partial(_dV, c0=method.c0)

    def initialize():
        xi = cv(helpers.query(snapshot))
        bias = np.zeros((natoms, dim))
        return GaMDState(
            xi, bias, snapshot.positions,
            0.0, 0.0, 0.0,                     # count, mean, m2
            np.inf, -np.inf,                   # pot_min, pot_max
            0.0, 0.0,                          # E, k0 (set at init_steps)
            np.zeros(1),                       # proj (dV)
            0, 0,                              # ncalls, nsteps
        )

    def update(state, data):
        V = data.energy                        # scalar physical potential energy
        Fphys = data.forces                    # (natoms, 3) unbiased physical force
        is_new = position_guard(state.prev_positions, data.positions)
        ncalls = state.ncalls + 1
        nsteps = state.nsteps + np.where(is_new, 1, 0)

        # --- collect energy statistics once per real step, while nsteps < equil_steps ---
        do_stat = is_new & (nsteps < equil_steps)
        count_t = state.count + 1.0
        mean_t, m2_t, _ = welford_var(count_t, state.mean, state.m2, V)
        count = np.where(do_stat, count_t, state.count)
        mean = np.where(do_stat, mean_t, state.mean)
        m2 = np.where(do_stat, m2_t, state.m2)
        pot_min = np.where(do_stat, np.minimum(state.pot_min, V), state.pot_min)
        pot_max = np.where(do_stat, np.maximum(state.pot_max, V), state.pot_max)

        # --- E, k0: recompute from init_steps through equil_steps, then freeze ---
        var = np.where(count > 2, m2 / count, 0.0)
        vstd = np.sqrt(np.maximum(var, 1e-12))
        E_new, k0_new = _params(pot_min, pot_max, mean, vstd, param)
        recompute = (nsteps >= init_steps) & (nsteps < equil_steps)
        E = np.where(recompute, E_new, state.E)
        k0 = np.where(recompute, k0_new, state.k0)

        # --- boost: on once nsteps >= init_steps; f'(V) gates itself at V<E (amd/gamd) ---
        boosting = nsteps >= init_steps
        coeff = np.where(boosting, _fprime(V, E, k0, pot_min, pot_max), 0.0)
        bias = coeff * Fphys                   # f'(V) * F_phys (added; no extra negation)
        proj = np.where(boosting, _dV(V, E, k0, pot_min, pot_max), 0.0).reshape(1)

        xi = cv(data)
        return GaMDState(xi, bias, data.positions, count, mean, m2, pot_min, pot_max,
                         E, k0, proj, ncalls, nsteps)

    return snapshot, initialize, generalize(update, helpers)
