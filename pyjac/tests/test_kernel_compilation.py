import os
import filecmp
from ..core.rate_subs import write_specrates_kernel
from . import TestClass
from ..loopy.loopy_utils import loopy_options

class SubTest(TestClass):
    def test_compile_specrates_knl(self):
        kgen = write_specrates_kernel(self.store.build_dir,
                {'conp' : self.store.conp_eqs, 'conv' : self.store.conv_eqs},
                self.store.reacs, self.store.specs,
                loopy_options(lang='opencl',
                    width=None, depth=None, ilp=False,
                    unr=None, order='C', platform='CPU'))