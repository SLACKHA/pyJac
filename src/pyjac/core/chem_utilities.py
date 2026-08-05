"""Module containing element dict, species and reaction classes, and constants.

"""

# Standard libraries
import functools
import math

import cantera as ct
import numpy as np

__all__ = ['RU', 'RUC', 'RU_JOUL', 'PA', 'get_elem_wt',
           'ReacInfo', 'SpecInfo', 'calc_spec_smh']

# universal gas constants, SI units
RU = ct.gas_constant  # J/(kmole * K)
RU_JOUL = ct.gas_constant / 1000.
RUC = (RU / 4.18400)  # cal/(mole * K)

# pressure of one standard atmosphere [Pa]
PA = ct.one_atm


class CommonEqualityMixin:
    """Base class for `ReacInfo` and `SpecInfo` classes for equality comparison
    """
    def __eq__(self, other):
        try:
            for key, value in self.__dict__.items():
                if key not in other.__dict__:
                    return False
                if isinstance(value, np.ndarray):
                    if not np.array_equal(value, other.__dict__[key]):
                        return False
                elif isinstance(value, list):
                    if not all([any(x == y for y in other.__dict__[key]) for x in value]):
                        return False
                elif value != other.__dict__[key]:
                    return False
            return True
        except Exception as e:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)


#: Conventional mass numbers for radioelements that Cantera declines to weigh,
#: since they have no stable isotopes. Retained from pyJac's original table so
#: that mechanisms naming them still parse.
_UNSTABLE_ELEM_WT = {
    'tc': 99.0, 'pm': 145.0, 'po': 210.0, 'at': 210.0, 'rn': 222.0,
    'fr': 223.0, 'ra': 226.0, 'ac': 227.0, 'pa': 231.0, 'np': 237.0,
    'pu': 242.0, 'am': 243.0, 'cm': 247.0, 'bk': 249.0, 'cf': 251.0,
    'es': 254.0, 'fm': 253.0,
}

#: Chemkin mechanisms use D for deuterium and E for the electron; Cantera
#: defines neither as an element.
_EXTRA_ELEM_WT = {'d': 2.014102, 'e': 5.4857990907e-4}


@functools.lru_cache(maxsize=1)
def _base_elem_wt():
    """Build the element weight table once; callers get their own copy."""
    elem_wt = {}
    for symbol in ct.Element.element_symbols:
        try:
            elem_wt[symbol.lower()] = ct.Element(symbol).weight
        except ct.CanteraError:
            continue

    elem_wt.update(_UNSTABLE_ELEM_WT)
    elem_wt.update(_EXTRA_ELEM_WT)
    return elem_wt


def get_elem_wt():
    """Returns dict with element names and atomic weights [kg/kmol].

    Weights come from Cantera, so that a mechanism read through the Chemkin
    parser and the same mechanism read through Cantera describe identical
    species masses.

    A fresh dictionary is returned on each call: callers overwrite entries
    when a mechanism declares its own atomic weights, and those overrides must
    not leak into the next mechanism read.

    Attributes
    ----------
    None

    Returns
    -------
    elem_wt : dict
        Dictionary with element name keys and atomic weight [kg/kmol] values.
    """
    return dict(_base_elem_wt())


class ReacInfo(CommonEqualityMixin):
    """Reaction class.

    Contains all information about a single reaction.

    Attributes
    ----------
    rev : bool
        True if reversible reaction, False if irreversible.
    reactants : list of str
        List of reactant species names.
    reac_nu : list of int/float
        List of reactant stoichiometric coefficients, either int or float.
    products : list of str
        List of product species names.
    prod_nu : list of int/float
        List of product stoichiometric coefficients, either int or float.
    A : float
        Arrhenius pre-exponential coefficient.
    b : float
        Arrhenius temperature exponent.
    E : float
        Arrhenius activation energy.
    rev_par : list of float, optional
        List of reverse Arrhenius coefficients (default empty).
    dup : bool, optional
        Duplicate reaction flag (default False).
    thd : bool, optional
        Third-body reaction flag (default False).
    thd_body : list of list of [str, float], optional
        List of third body names and efficiencies (default empty).
    pdep : bool, optional
        Pressure-dependence flag (default False).
    pdep_sp : str, optional
        Name of specific third-body or 'M' (default '').
    low : list of float, optional
        List of low-pressure-limit Arrhenius coefficients (default empty).
    high : list of float, optional
        List of high-pressure-limit Arrhenius coefficients (default empty).
    troe : bool, optional
        Troe pressure-dependence formulation flag (default False).
    troe_par : list of float, optional
        List of Troe formulation constants (default empty).
    sri : bool, optional
        SRI pressure-dependence formulation flag (default False).
    sri_par : list of float, optional
        List of SRI formulation constants (default empty).

    Notes
    -----
    `rev` does not require `rev_par`; if no explicit coefficients, the
    reverse reaction rate will be calculated through the equilibrium
    constant.
    Only one of [`low`,`high`] can be defined.
    If `troe` and `sri` are both False, then the Lindemann is assumed.

    """

    def __init__(self, rev, reactants, reac_nu, products, prod_nu, A, b, E):
        self.reac = reactants
        self.reac_nu = reac_nu
        self.prod = products
        self.prod_nu = prod_nu

        ## Arrhenius coefficients
        # pre-exponential factor [m, kmol, s]
        self.A = A
        # Temperature exponent [-]
        self.b = b
        # Activation energy, stored as activation temperature [K]
        self.E = E

        # reversible reaction properties
        self.rev = rev
        self.rev_par = []  # reverse A, b, E

        # duplicate reaction
        self.dup = False

        # third-body efficiencies
        self.thd_body = False
        self.thd_body_eff = []  # in pairs with species and efficiency

        # pressure dependence
        self.pdep = False
        self.pdep_sp = ''
        self.low = []
        self.high = []

        self.troe = False
        self.troe_par = []

        self.sri = False
        self.sri_par = []

        # Parameters for pressure-dependent reaction parameterized by
        # bivariate Chebyshev polynomial in temperature and pressure.
        self.cheb = False
        # Number of temperature values over which fit computed.
        self.cheb_n_temp = 0
        # Number of pressure values over which fit computed.
        self.cheb_n_pres = 0
        # Pressure limits for Chebyshev fit [Pa]
        self.cheb_plim = [0.001 * PA, 100. * PA]
        # Temperature limits for Chebyshev fit [K]
        self.cheb_tlim = [300., 2500.]
        # 2D array of Chebyshev fit coefficients
        self.cheb_par = None

        # Parameters for pressure-dependent reaction parameterized by
        # logarithmically interpolating between Arrhenius rate expressions at
        # various pressures.
        self.plog = False
        # List of arrays with [pressure [Pa], A, b, E]
        self.plog_par = None


class SpecInfo(CommonEqualityMixin):
    """Species class.

    Contains all information about a single species.

    Attributes
    ----------
    name : str
        Name of species.
    elem : list of list of [str, float]
        Elemental composition in [element, number] pairs.
    mw : float
        Molecular weight.
    hi : list of float
        High-temperature range NASA thermodynamic coefficients.
    lo : list of float
        Low-temperature range NASA thermodynamic coefficients.
    Trange : list of float
        Temperatures defining ranges of thermodynamic polynomial fits
        (low, middle, high), default ([300, 1000, 5000]).

    """

    def __init__(self, name):
        self.name = name

        # elemental composition
        self.elem = []
        # molecular weight [kg/kmol]
        self.mw = 0.0
        # high-temp range thermodynamic coefficients
        self.hi = np.zeros(7)
        # low-temp range thermodynamic coefficients
        self.lo = np.zeros(7)
        # temperature [K] range for thermodynamic coefficients
        self.Trange = [300.0, 1000.0, 5000.0]


def calc_spec_smh(T, specs):
    """Calculate standard-state entropies minus enthalpies for all species.

    Parameters
    ----------
    T : float
        Temperature of gas mixture.
    specs : list of SpecInfo
        List of species.

    Returns
    -------
    spec_smh : list of float
        List of species' standard-state entropies minus enthalpies.

    """

    spec_smh = []

    Tlog = math.log(T)
    T2 = T * T
    T3 = T2 * T
    T4 = T3 * T

    Thalf = T / 2.0
    T2 = T2 / 6.0
    T3 = T3 / 12.0
    T4 = T4 / 20.0

    for sp in specs:
        if T <= sp.Trange[1]:
            smh = (sp.lo[0] * (Tlog - 1.0) + sp.lo[1] * Thalf + sp.lo[2] *
                   T2 + sp.lo[3] * T3 + sp.lo[4] * T4 - (sp.lo[5] / T) +
                   sp.lo[6]
                   )
        else:
            smh = (sp.hi[0] * (Tlog - 1.0) + sp.hi[1] * Thalf + sp.hi[2] *
                   T2 + sp.hi[3] * T3 + sp.hi[4] * T4 - (sp.hi[5] / T) +
                   sp.hi[6]
                   )

        spec_smh.append(smh)

    return (spec_smh)
