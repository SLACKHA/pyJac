"""Module containing element dict, species and reaction classes, and constants.

"""

# Python 2 compatibility
from __future__ import division

# Standard libraries
import math
import numpy as np

__all__ = ['RU', 'RUC', 'RU_JOUL', 'PA', 'get_elem_wt',
           'ReacInfo', 'SpecInfo', 'calc_spec_smh']

# universal gas constants, SI units
RU = 8314.4621  # J/(kmole * K)
RU_JOUL = 8.3144621
RUC = (RU / 4.18400)  # cal/(mole * K)

# Avogadro's number
AVAG = 6.0221367e23

# pressure of one standard atmosphere [Pa]
PA = 101325.0


class CommonEqualityMixin(object):
    """Base class for `ReacInfo` and `SpecInfo` classes for equality comparison
    """
    def __eq__(self, other):
        try:
            for key, value in self.__dict__.items():
                if not key in other.__dict__:
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


def get_elem_wt():
    """Returns dict with built-in element names and atomic weights [kg/kmol].

    Attributes
    ----------
    None

    Returns
    -------
    elem_wt : dict
        Dictionary with element name keys and atomic weight [kg/kmol] values.
    """
    elem_wt = dict([
        ('h', 1.00794), ('he', 4.00260), ('li', 6.93900),
        ('be', 9.01220), ('b', 10.81100), ('c', 12.0110),
        ('n', 14.00674), ('o', 15.99940), ('f', 18.99840),
        ('ne', 20.18300), ('na', 22.98980), ('mg', 24.31200),
        ('al', 26.98150), ('si', 28.08600), ('p', 30.97380),
        ('s', 32.06400), ('cl', 35.45300), ('ar', 39.94800),
        ('k', 39.10200), ('ca', 40.08000), ('sc', 44.95600),
        ('ti', 47.90000), ('v', 50.94200), ('cr', 51.99600),
        ('mn', 54.93800), ('fe', 55.84700), ('co', 58.93320),
        ('ni', 58.71000), ('cu', 63.54000), ('zn', 65.37000),
        ('ga', 69.72000), ('ge', 72.59000), ('as', 74.92160),
        ('se', 78.96000), ('br', 79.90090), ('kr', 83.80000),
        ('rb', 85.47000), ('sr', 87.62000), ('y', 88.90500),
        ('zr', 91.22000), ('nb', 92.90600), ('mo', 95.94000),
        ('tc', 99.00000), ('ru', 101.07000), ('rh', 102.90500),
        ('pd', 106.40000), ('ag', 107.87000), ('cd', 112.40000),
        ('in', 114.82000), ('sn', 118.69000), ('sb', 121.75000),
        ('te', 127.60000), ('i', 126.90440), ('xe', 131.30000),
        ('cs', 132.90500), ('ba', 137.34000), ('la', 138.91000),
        ('ce', 140.12000), ('pr', 140.90700), ('nd', 144.24000),
        ('pm', 145.00000), ('sm', 150.35000), ('eu', 151.96000),
        ('gd', 157.25000), ('tb', 158.92400), ('dy', 162.50000),
        ('ho', 164.93000), ('er', 167.26000), ('tm', 168.93400),
        ('yb', 173.04000), ('lu', 174.99700), ('hf', 178.49000),
        ('ta', 180.94800), ('w', 183.85000), ('re', 186.20000),
        ('os', 190.20000), ('ir', 192.20000), ('pt', 195.09000),
        ('au', 196.96700), ('hg', 200.59000), ('tl', 204.37000),
        ('pb', 207.19000), ('bi', 208.98000), ('po', 210.00000),
        ('at', 210.00000), ('rn', 222.00000), ('fr', 223.00000),
        ('ra', 226.00000), ('ac', 227.00000), ('th', 232.03800),
        ('pa', 231.00000), ('u', 238.03000), ('np', 237.00000),
        ('pu', 242.00000), ('am', 243.00000), ('cm', 247.00000),
        ('bk', 249.00000), ('cf', 251.00000), ('es', 254.00000),
        ('fm', 253.00000), ('d', 2.01410), ('e', 5.48578e-4)
    ])
    return elem_wt


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
