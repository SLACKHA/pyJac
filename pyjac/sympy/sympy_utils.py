"""
This module contains a number of commonly used tools that help translate
Sympy equations to loopy code
"""

from sympy import Symbol

def sanitize(eqn, subs={}, symlist={}):
    """
    Sanitizes a sympy equation for insertion into loopy code:

        * All free_symbols are replaced with simple variables so as not to deal with oddities of sympy printing
        * Simpler name substitutions may be specified in symlist
        * Complex substitutions may be specified in the subs dict

    Parameters
    ----------
    subs : dict of `sympy.Symbol` -> `sympy.Symbol`
        a list of complex substitions do perform
    symlist : dict of str -> `sympy.Symbol`
        A list of simpler symbol names to substitute into the equation

    """
    #first we replace values with regular symbols for easy working
    #if an equivalent value is not found in the symlist the str conversion of the symbol will be used
    indexed = [(x, next((sym for name, sym in symlist.items()
                if name == str(x)), Symbol(str(x)))) for x in eqn.free_symbols]
    eqn = eqn.subs(indexed)
    #next we do any complex user specified substitution
    eqn = eqn.subs([(k, v) for k, v in subs.items()])
    return eqn