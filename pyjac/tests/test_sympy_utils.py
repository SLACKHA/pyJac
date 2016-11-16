#local imports
from . import TestClass
from ..sympy import sympy_utils as sp_util

#modules
import sympy as sp

class SubTest(TestClass):
    def test_subs(self):
        #make symbols
        A = sp.IndexedBase('A')
        B = sp.IndexedBase('B')
        i = sp.Idx('i')

        expr = A[i] + B[i]

        A_dummy = sp.Symbol('A[i]')
        B_dummy = sp.Symbol('B[i]')

        expr_san = sp_util.sanitize(expr)
        assert len(expr.free_symbols.intersection({A_dummy, B_dummy, i})) ==\
                    len(expr.free_symbols)
