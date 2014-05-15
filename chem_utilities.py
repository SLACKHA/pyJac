import math
from __future__ import division

# universal gas constants, cgs units
RU = 8.314510e7 # erg/(mole * K)
RUC = 1.9858775 # cal/(mole * K)
RU_JOUL = 8.314510e0
# pressure of one standard atmosphere, dynes/cm^2
PA = 1.01325e6

# element atomic weights
elem_mw = dict( [ 
    ('h',   1.00797),  ('he',   4.00260), ('li',   6.93900), ('be',   9.01220), 
    ('b',  10.81100),  ('c',   12.01115), ('n',   14.00670), ('o',   15.99940), 
    ('f',  18.99840),  ('ne',  20.18300), ('na',  22.98980), ('mg',  24.31200),
    ('al', 26.98150),  ('si',  28.08600), ('p',   30.97380), ('s',   32.06400), 
    ('cl', 35.45300),  ('ar',  39.94800), ('k',   39.10200), ('ca',  40.08000), 
    ('sc', 44.95600),  ('ti',  47.90000), ('v',   50.94200), ('cr',  51.99600), 
    ('mn', 54.93800),  ('fe',  55.84700), ('co',  58.93320), ('ni',  58.71000), 
    ('cu', 63.54000),  ('zn',  65.37000), ('ga',  69.72000), ('ge',  72.59000), 
    ('as', 74.92160),  ('se',  78.96000), ('br',  79.90090), ('kr',  83.80000), 
    ('rb', 85.47000),  ('sr',  87.62000), ('y',   88.90500), ('zr',  91.22000),
    ('nb', 92.90600),  ('mo',  95.94000), ('tc',  99.00000), ('ru', 101.07000),
    ('rh', 102.90500), ('pd', 106.40000), ('ag', 107.87000), ('cd', 112.40000),
    ('in', 114.82000), ('sn', 118.69000), ('sb', 121.75000), ('te', 127.60000),
    ('i',  126.90440), ('xe', 131.30000), ('cs', 132.90500), ('ba', 137.34000),
    ('la', 138.91000), ('ce', 140.12000), ('pr', 140.90700), ('nd', 144.24000),
    ('pm', 145.00000), ('sm', 150.35000), ('eu', 151.96000), ('gd', 157.25000),
    ('tb', 158.92400), ('dy', 162.50000), ('ho', 164.93000), ('er', 167.26000),
    ('tm', 168.93400), ('yb', 173.04000), ('lu', 174.99700), ('hf', 178.49000),
    ('ta', 180.94800), ('w',  183.85000), ('re', 186.20000), ('os', 190.20000),
    ('ir', 192.20000), ('pt', 195.09000), ('au', 196.96700), ('hg', 200.59000),
    ('tl', 204.37000), ('pb', 207.19000), ('bi', 208.98000), ('po', 210.00000),
    ('at', 210.00000), ('rn', 222.00000), ('fr', 223.00000), ('ra', 226.00000),
    ('ac', 227.00000), ('th', 232.03800), ('pa', 231.00000), ('u',  238.03000),
    ('np', 237.00000), ('pu', 242.00000), ('am', 243.00000), ('cm', 247.00000),
    ('bk', 249.00000), ('cf', 251.00000), ('es', 254.00000), ('fm', 253.00000),
    ('d',    2.01410), ('e', 5.48578e-4) ] )

# dict for any new element definitions
elem_mw_new = dict()

class ReacInfo:
    """Class for reaction information"""
    
    def __init__(self, rev, reactants, reac_nu, products, prod_nu, A, b, E):
        self.reac = reactants
        self.reac_nu = reac_nu
        self.prod = products
        self.prod_nu = prod_nu
        
        # Arrhenius coefficients
        self.A = A
        self.b = b
        self.E = E
        
        # reversible reaction properties
        self.rev = rev
        self.rev_par = [] # reverse A, b, E
        
        # duplicate reaction
        self.dup = False
        
        # third-body efficiencies
        self.thd = False
        self.thd_body = [] # in pairs with species and efficiency
        
        # pressure dependence
        self.pdep = False
        self.pdep_sp = ''
        self.low = []
        self.high = []
        
        self.troe = False
        self.troe_par = []
        
        self.sri = False
        self.sri_par = []


class SpecInfo:
    """Class for species information"""
    
    def __init__(self, name):
        self.name = name
        
        # elemental composition
        self.elem = []
        # molecular weight
        self.mw = 0.0
        # high-temp range thermodynamic coefficients
        self.hi = [0.0 for j in range(7)]
        # low-temp range thermodynamic coefficients
        self.lo = [0.0 for j in range(7)]
        # temperature range for thermodynamic coefficients
        self.Trange = [0.0, 0.0, 0.0]


def calc_spec_smh(T, specs):
    """Calculate standard-state entropies minus enthalpies for all species
    
    Input
    T: temperature
    specs: list of species (SpecInfo class)
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
            smh = sp.lo[0] * (Tlog - 1.0) + sp.lo[1] * Thalf + sp.lo[2] * T2 + \
                  sp.lo[3] * T3 + sp.lo[4] * T4 - (sp.lo[5] / T) + sp.lo[6]
        else:
            smh = sp.hi[0] * (Tlog - 1.0) + sp.hi[1] * Thalf + sp.hi[2] * T2 + \
                  sp.hi[3] * T3 + sp.hi[4] * T4 - (sp.hi[5] / T) + sp.hi[6]
        
        spec_smh.append(smh)
    
    return(spec_smh)

