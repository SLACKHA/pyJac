"""A reference Jacobian for pyJac's state vector, built from Cantera.

pyJac's state vector is :math:`\\Phi = \\{T, Y_1 \\ldots Y_{N-1}\\}` at constant
pressure, with one species eliminated by :math:`Y_N = 1 - \\sum Y_k`. Cantera
exposes analytic kinetics derivatives with respect to temperature and species
*concentrations*, so the chain rule here converts those into derivatives with
respect to :math:`\\Phi`.

Finite differencing the *full* Jacobian is not accurate enough to validate
pyJac against -- that was a finding of the pyJac paper, and the species block
is where it fails. Two narrower uses survive that objection and both appear
below: `finite_difference_jacobian` checks this module's own chain rule, where
both sides evaluate the same right-hand side so the comparison tests calculus
rather than chemistry; and `temperature_derivative_by_extrapolation` supplies
one well-conditioned scalar derivative more accurately than Cantera can.
"""

import cantera as ct
import numpy as np


def species_cp_slopes(gas):
    """Returns d(cp_k)/dT for each species, mass specific.

    Evaluated from the NASA polynomial coefficients, where cp/R is a quartic in
    temperature, so the result is exact.
    """
    temperature = gas.T
    slopes = np.zeros(gas.n_species)
    for k in range(gas.n_species):
        thermo = gas.species(k).thermo
        coeffs = thermo.coeffs
        if thermo.n_coeffs != 15:
            raise NotImplementedError(
                f'{gas.species_name(k)} does not use a 7-coefficient NASA '
                f'polynomial; this reference cannot differentiate it'
            )
        # coeffs is [T_mid, high range (7), low range (7)]
        poly = coeffs[1:8] if temperature > coeffs[0] else coeffs[8:15]
        dcp_over_r = (
            poly[1]
            + 2.0 * poly[2] * temperature
            + 3.0 * poly[3] * temperature**2
            + 4.0 * poly[4] * temperature**3
        )
        slopes[k] = ct.gas_constant * dcp_over_r / gas.molecular_weights[k]
    return slopes


def state_vector(gas, order):
    """Returns pyJac's state vector for the gas' current state."""
    return np.concatenate(([gas.T], gas.Y[order[:-1]]))


def apply_state(gas, temperature, partial_y, pressure, order):
    """Sets the gas from a pyJac state vector.

    ``partial_y`` holds the mass fractions of ``order[:-1]``; the species in
    ``order[-1]`` takes up the remainder.
    """
    mass_fractions = np.zeros(gas.n_species)
    mass_fractions[order[:-1]] = partial_y
    mass_fractions[order[-1]] = 1.0 - partial_y.sum()
    gas.TPY = temperature, pressure, mass_fractions


def derivative(gas, temperature, partial_y, pressure, order):
    """Returns dPhi/dt: the right-hand side pyJac's Jacobian differentiates."""
    apply_state(gas, temperature, partial_y, pressure, order)

    production = gas.net_production_rates
    density = gas.density_mass

    temperature_dot = -np.dot(gas.partial_molar_enthalpies, production) / (
        density * gas.cp_mass
    )
    mass_fraction_dot = gas.molecular_weights * production / density
    return np.concatenate(([temperature_dot], mass_fraction_dot[order[:-1]]))


def analytic_jacobian(gas, temperature, partial_y, pressure, order):
    """Returns dPhi_dot/dPhi from Cantera's analytic kinetics derivatives.

    Parameters
    ----------
    gas : `cantera.Solution`
        Mechanism to evaluate. Its state is overwritten.
    temperature : float
        Temperature in K.
    partial_y : `numpy.ndarray`
        Mass fractions of every species except the eliminated one, ordered by
        ``order[:-1]``.
    pressure : float
        Pressure in Pa, held constant.
    order : `numpy.ndarray`
        Cantera species indices in pyJac's order. The last entry is the
        eliminated species.

    Returns
    -------
    `numpy.ndarray`
        Square matrix of side ``gas.n_species``, indexed as pyJac's Jacobian.

    """
    apply_state(gas, temperature, partial_y, pressure, order)

    n_species = gas.n_species
    kept, eliminated = order[:-1], order[-1]

    weights = gas.molecular_weights
    density = gas.density_mass
    mean_weight = gas.mean_molecular_weight
    cp = gas.cp_mass
    production = gas.net_production_rates
    concentrations = gas.concentrations
    molar_enthalpy = gas.partial_molar_enthalpies
    molar_cp = gas.partial_molar_cp
    heat_release = np.dot(molar_enthalpy, production)

    dproduction_dconc = np.asarray(gas.net_production_rates_ddCi)
    dproduction_dtemperature = np.asarray(gas.net_production_rates_ddT)

    jacobian = np.zeros((n_species, n_species))

    def fill_column(column, ddensity, dproduction, dcp):
        """Writes one column, given how density, rates and cp respond to it."""
        dheat = np.dot(molar_enthalpy, dproduction)
        if column == 0:
            dheat += np.dot(molar_cp, production)
        jacobian[0, column] = -dheat / (density * cp) + heat_release / (
            density * cp
        ) * (ddensity / density + dcp / cp)
        jacobian[1:, column] = (
            weights[kept] * dproduction[kept] / density
            - weights[kept] * production[kept] * ddensity / density**2
        )

    # Temperature column. At constant pressure and composition the density and
    # every concentration fall as 1/T, on top of the explicit rate dependence.
    ddensity_dtemperature = -density / temperature
    dconc_dtemperature = -concentrations / temperature
    fill_column(
        0,
        ddensity_dtemperature,
        dproduction_dtemperature + dproduction_dconc @ dconc_dtemperature,
        np.dot(gas.Y, species_cp_slopes(gas)),
    )

    # Species columns. Raising one mass fraction lowers the eliminated species
    # by the same amount, shifting the mean molecular weight and so the density
    # and every concentration.
    species_cp = molar_cp / weights
    for position, species in enumerate(kept):
        dmean_weight = -(mean_weight**2) * (
            1.0 / weights[species] - 1.0 / weights[eliminated]
        )
        ddensity = density * dmean_weight / mean_weight

        dconc = concentrations * ddensity / density
        dconc[species] += density / weights[species]
        dconc[eliminated] -= density / weights[eliminated]

        fill_column(
            position + 1,
            ddensity,
            dproduction_dconc @ dconc,
            species_cp[species] - species_cp[eliminated],
        )

    return jacobian


def temperature_derivative_by_extrapolation(
    gas, temperature, partial_y, pressure, order, step=0.01
):
    """Returns d(dT/dt)/dT by Richardson-extrapolated central difference.

    A more accurate reference for this one entry than `analytic_jacobian`.
    Cantera has no analytic temperature derivative for PLOG or Chebyshev rates
    and finite-differences them internally, which caps the analytic reference
    at a few parts in 1e7 for a mechanism carrying those. Tightening Cantera's
    own step does not fix it: the error is U-shaped in the step size and its
    minimum sits at a different place for each state.

    Differencing here instead is sound because this is a single well-scaled
    scalar derivative of a smooth function, not the stiff species block the
    pyJac paper found finite differences unequal to. Central differencing is
    second order, so evaluating at ``step`` and ``2 * step`` and extrapolating
    cancels the leading error term, giving agreement to ~2e-10 -- the same
    floating-point floor the rest of the matrix reaches.
    """

    def central(width):
        forward = derivative(gas, temperature + width, partial_y, pressure, order)
        backward = derivative(gas, temperature - width, partial_y, pressure, order)
        return (forward[0] - backward[0]) / (2.0 * width)

    coarse, fine = central(2.0 * step), central(step)
    return (4.0 * fine - coarse) / 3.0


def finite_difference_jacobian(gas, temperature, partial_y, pressure, order):
    """Returns a central-difference Jacobian of the same right-hand side.

    Only accurate enough to confirm the chain rule in `analytic_jacobian`; not
    a reference for validating pyJac.
    """
    size = len(partial_y) + 1
    jacobian = np.zeros((size, size))

    step = temperature * 1.0e-6
    forward = derivative(gas, temperature + step, partial_y, pressure, order)
    backward = derivative(gas, temperature - step, partial_y, pressure, order)
    jacobian[:, 0] = (forward - backward) / (2.0 * step)

    for position in range(size - 1):
        step = max(abs(partial_y[position]), 1.0e-8) * 1.0e-6
        shifted = partial_y.copy()
        shifted[position] += step
        forward = derivative(gas, temperature, shifted, pressure, order)
        shifted[position] -= 2.0 * step
        backward = derivative(gas, temperature, shifted, pressure, order)
        jacobian[:, position + 1] = (forward - backward) / (2.0 * step)

    return jacobian
