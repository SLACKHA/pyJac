from .sympy_addons import *

class MyImplicitSymbol(ImplicitSymbol):
    def _get_df(self, arg, wrt):
        if isinstance(arg, IndexedConc) and \
                    isinstance(wrt, MyIndexedFunc.MyIndexedFuncValue) and \
                    isinstance(wrt.base, IndexedConc):
            return self.__class__(self.base_str.format(
                    str(self.name), str(wrt)), args=self.functional_form)
        return self.__class__(self.base_str.format(
                str(self.name), str(arg)), args=self.functional_form)

class MyIndexedFunc(IndexedFunc):
    def _get_subclass(self, *args):
        return MyIndexedFunc.MyIndexedFuncValue(*args)

    class MyIndexedFuncValue(IndexedFunc.IndexedFuncValue):
        def _get_df(self, arg, wrt):
            if isinstance(arg, IndexedConc) and \
                    isinstance(wrt, MyIndexedFunc.MyIndexedFuncValue) and \
                    isinstance(wrt.base, IndexedConc):
                return self.base.__class__(self.base_str.format(
                        str(self.base), str(wrt)), self.functional_form)[self.indices]
            return super(MyIndexedFunc.MyIndexedFuncValue, self)._get_df(arg, wrt)

#some custom behaviour for concentrations
class IndexedConc(MyIndexedFunc):
    is_Real = True
    is_Positive = True
    is_Negative = False
    is_Number = True
    _diff_wrt = True
    def _eval_derivative(self, wrt):
        if isinstance(wrt, MyIndexedFunc.MyIndexedFuncValue) and \
            isinstance(wrt.base, IndexedConc):
            return S.One
        return S.Zero