import os
import filecmp
from ..core.rate_subs import write_specrates_kernel, write_chem_utils
from . import TestClass
from ..loopy.loopy_utils import loopy_options
from ..libgen import generate_library
from ..core.mech_auxiliary import write_mechanism_header
from optionloop import OptionLoop
from collections import OrderedDict

class SubTest(TestClass):
    def test_compile_specrates_knl(self):
        opts = loopy_options(lang='opencl',
                    width=None, depth=None, ilp=False,
                    unr=None, order='C', platform='CPU')
        eqs = {'conp' : self.store.conp_eqs, 'conv' : self.store.conv_eqs}

        oploop = OptionLoop(OrderedDict([
            ('conp', [True, False]),
            ('shared', [True, False])]))

        build_dir = self.store.build_dir
        out_dir = os.path.realpath(os.path.join(self.store.build_dir,
                        os.pardir, 'lib'))
        for state in oploop:
            conp = state['conp']
            shared = state['shared']

            kgen = write_specrates_kernel(eqs, self.store.reacs, self.store.specs, opts,
                conp=conp)
            kgen2 = write_chem_utils(self.store.specs, eqs, opts)
            #add deps
            kgen.add_depencencies([kgen2])
            #generate
            kgen.generate(build_dir)
            #write header
            write_mechanism_header(build_dir, opts.lang, self.store.specs, self.store.reacs)
            #compile
            generate_library(opts.lang, build_dir, obj_dir=None,
                         out_dir=out_dir, shared=shared,
                         finite_difference=False, auto_diff=False)

    def test_specrates_pywrap(self):
        pass