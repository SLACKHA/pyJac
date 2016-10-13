from enum import Enum

class reaction_type(Enum):
    elementary = 1
    thd = 2
    fall = 3
    chem = 4
    plog = 5
    cheb = 6

class thd_body_type(Enum):
    none = 0
    mix = 1
    species = 2
    unity = 3

class falloff_form(Enum):
    none = 0
    lind = 1
    troe = 2
    sri = 3

class reversible_type(Enum):
    non_reversible = 1
    explicit = 2
    non_explicit = 3