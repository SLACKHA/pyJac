import os
from ..core.sympy_interpreter import load_equations, enum_from_str
from ..core.reaction_types import *

script_dir = os.path.abspath(os.path.dirname(__file__))
def test_sympy_exists():
    core_path = os.path.normpath(os.path.join(script_dir, os.pardir, 'core'))
    assert os.path.isfile(os.path.join(core_path, 'conp_derivation.sympy'))
    assert os.path.isfile(os.path.join(core_path, 'conv_derivation.sympy'))

def test_load_runs():
    var, eqns = load_equations(True)
    var, eqns = load_equations(False)

def test_enum_from_str():
    from nose.tools import set_trace; set_trace()
    assert enum_from_str('reversible_type.explicit') == reversible_type.explicit
    try:
        assert enum_from_str('reversible_type.not_real')
        assert False
    except:
        pass

def test_file_structure():
    def __subtest(conp):
        var, eqns = load_equations(conp, True)
        assert all(key in var for key in eqns.keys())

    __subtest(True)
    __subtest(False)