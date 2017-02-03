import filecmp
import os

from ..core.mech_auxiliary import write_mechanism_header
from . import TestClass

class SubTest(TestClass):
    def test_write_mechanism_header(self):
        script_dir = self.store.script_dir
        build_dir = self.store.build_dir
        write_mechanism_header(build_dir, 'opencl', self.store.specs, self.store.reacs)
        assert filecmp.cmp(os.path.join(build_dir, 'mechanism.oclh'),
                        os.path.join(script_dir, 'blessed', 'mechanism.oclh'))