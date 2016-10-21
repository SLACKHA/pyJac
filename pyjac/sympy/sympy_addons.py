from sympy.core.symbol import Symbol
from sympy.tensor.indexed import Idx, IndexedBase, Indexed
from sympy.concrete import Product
from sympy.core.compatibility import is_sequence
from sympy.core.singleton import S
from sympy.core.add import Add
from sympy.core.function import Derivative
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.simplify.simplify import simplify

base_str_total = r'\frac{{\text{{d}} {} }}{{\text{{d}} {} }}'
base_str_partial = r'\frac{{\partial {} }}{{\partial {} }}'

class ImplicitSymbol(Symbol):
    def __new__(cls, name, args, **assumptions):
        obj = Symbol.__new__(cls, name, **assumptions)
        obj.functional_form = args
        obj.base_str = base_str_total if len(obj._get_iter_func()) == 1\
                            else base_str_partial 
        return obj

    def _get_iter_func(self):
        funcof = self.functional_form
        if not funcof:
            return []
        if not hasattr(self.functional_form, '__iter__'):
            funcof = [self.functional_form]

        return funcof

    def _eval_subs(self, old, new):
        if old == self:
            return new
        funcof = self._get_iter_func()
        for a in funcof:
            if a.has(old):
                new_func = [x if x != a else a.subs(old, new) 
                                for x in funcof]
                return self.__class__(str(self), new_func)
        return self

    @property
    def free_symbols(self):
        return super(ImplicitSymbol, self).free_symbols.union(*[
            x.free_symbols for x in self._get_iter_func()])

    def _eval_diff(self, wrt, **kw_args):
            return self._eval_derivative(wrt)

    def _get_df(self, a, wrt):
        return ImplicitSymbol(self.base_str.format(
                str(self.name), str(a)), args=self.functional_form)

    def _eval_derivative(self, wrt):
        if self == wrt:
            return S.One
        else:
            funcof = self._get_iter_func()
            i = 0
            l = []
            
            for a in funcof:
                i += 1
                da = a.diff(wrt)
                if da is S.Zero:
                    continue
                df = self._get_df(a, wrt)
                l.append(df * da)
            return Add(*l)

class IndexedFunc(IndexedBase):
    def __new__(cls, label, args, shape=None, **kw_args):
        obj = IndexedBase.__new__(cls, label, shape=shape, **kw_args)
        obj.functional_form = args
        return obj

    def _eval_simplify(self, ratio=1.7, measure=None):
        return self.__class__(self.label,
            *[simplify(x, ratio=ratio, measure=measure)
                         for x in self._get_iter_func()])

    def _get_iter_func(self):
        funcof = self.functional_form
        if not hasattr(self.functional_form, '__iter__'):
            funcof = [self.functional_form]

        return funcof

    @property
    def free_symbols(self):
        return set([self]).union(*[
            x.free_symbols for x in self._get_iter_func()])

    def _get_subclass(self, *args):
        return IndexedFunc.IndexedFuncValue(*args)

    class IndexedFuncValue(Indexed):
        def __new__(cls, base, *args):
            functional_form = args[0]
            obj = Indexed.__new__(cls, base, *args)
            obj.functional_form = functional_form
            obj.base_str = base_str_total if len(obj._get_iter_func()) == 1 else base_str_partial 
            return obj

        @property
        def indices(self):
            return self.args[2:]

        def _eval_simplify(self, ratio=1.7, measure=None):
            return self.__class__(
                        self.base,
                        *[simplify(x, ratio=ratio, measure=measure)
                                 for x in self._get_iter_func()])

        def _eval_subs(self, old, new):
            if self == old:
                return new
            if any(x.has(old) for x in self._get_iter_func()):
                return self.__class__(self.base,
                tuple(x.subs(old, new) for x in self._get_iter_func()),
                *self.indices)
            elif any(x.has(old) for x in self.indices):
                return self.__class__(self.base,
                self.functional_form,
                *tuple(x.subs(old, new) for x in self.indices))
            return self

        def _get_iter_func(self):
            funcof = self.functional_form
            if not hasattr(self.functional_form, '__iter__'):
                funcof = [self.functional_form]
            return funcof

        def _get_df(self, a, wrt):
            return self.base.__class__(self.base_str.format(
                    str(self.base), str(a)), args=self.functional_form)[self.indices]

        def _eval_diff(self, wrt, **kw_args):
            return self._eval_derivative(wrt)
        def _eval_derivative(self, wrt):
            if self == wrt:
                return S.One
            elif isinstance(wrt, IndexedFunc.IndexedFuncValue) and wrt.base == self.base:
                if len(self.indices) != len(wrt.indices):
                    msg = "Different # of indices: d({!s})/d({!s})".format(self,
                                                                           wrt)
                    raise IndexException(msg)
                elif self.functional_form != wrt.functional_form:
                    msg = "Different function form d({!s})/d({!s})".format(self.functional_form,
                                                                        wrt.functional_form)
                    raise IndexException(msg)
                result = S.One
                for index1, index2 in zip(self.indices, wrt.indices):
                    result *= KroneckerDelta(index1, index2)
                return result
            else:
                #f(x).diff(s) -> x.diff(s) * f.fdiff(1)(s)
                i = 0
                l = []
                funcof = self._get_iter_func()
                for a in funcof:
                    i += 1
                    da = a.diff(wrt)
                    if da is S.Zero:
                        continue
                    df = self._get_df(a, wrt)
                    l.append(df * da)
                return Add(*l)

        @property
        def free_symbols(self):
            return super(IndexedFunc.IndexedFuncValue, self).free_symbols.union(*[
            set([x]) if not isinstance(x, IndexedFunc.IndexedFuncValue) else
            x.free_symbols for x in self._get_iter_func()])

    def __getitem__(self, indices, **kw_args):
        if is_sequence(indices):
            # Special case needed because M[*my_tuple] is a syntax error.
            if self.shape and len(self.shape) != len(indices):
                raise IndexException("Rank mismatch.")
            return self._get_subclass(self,
                self.functional_form,
                *indices, **kw_args)
        else:
            if self.shape and len(self.shape) != 1:
                raise IndexException("Rank mismatch.")
            return self._get_subclass(self,
                self.functional_form,
                indices, **kw_args)