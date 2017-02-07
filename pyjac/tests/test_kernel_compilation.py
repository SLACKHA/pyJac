import os
import filecmp
from ..core.rate_subs import write_specrates_kernel, write_chem_utils
from . import TestClass
from ..loopy.loopy_utils import loopy_options
from ..libgen import generate_library

class SubTest(TestClass):
    def test_compile_specrates_knl(self):
        opts = loopy_options(lang='opencl',
                    width=None, depth=None, ilp=False,
                    unr=None, order='C', platform='CPU')
        eqs = {'conp' : self.store.conp_eqs, 'conv' : self.store.conv_eqs}
        kgen = write_specrates_kernel(eqs, self.store.reacs, self.store.specs, opts)
        kgen2 = write_chem_utils(self.store.specs, eqs, opts)
        #add dep
        kgen.add_depencencies([kgen2])

        kgen.generate(self.store.build_dir)

        generate_library(opts.lang, self.store.build_dir, obj_dir=None,
                     out_dir=None, shared=None,
                     finite_difference=False, auto_diff=False)