from ..sympy import load_equations
from ..core import read_mech
from ..core.reaction_types import reaction_type as rtype

#load some equations
conp_vars, conp_eqs = sp_interp.load_equations(True)

def test_match():
    #get the kf equations
    kf = next(x for x in conp_vars if str(x) == '{k_f}[i]')
    kf_eqs = [x for x in conp_eqs if x.has(kf)]
    kf_eqs = {key: (x, conp_eqs[x][key]) for x in kf_eqs for key in conp_eqs[x]}

    #load a mechanism
    elem, reac, specs = read_mech('test.inp')

    for reac in reacs:
        #test that it matches
        key = next(x for x in kf_eqs if reac.match(x))
        #test that the match is correct
        if reac.pdep:
            assert rtype.fall in key or rtype.chem in key
        elif reac.thd_body:
            assert rtype.thd in key
        elif reac.plog:
            assert rtype.plog in key
        elif reac.cheb:
            assert rtype.cheb in key
        else:
            assert rtype.elementary in key