import loopy as lp
import numpy as np
import pyopencl as cl
import re
from ..utils import check_lang

import os

class loopy_options(object):
    """
    Loopy Objects class

    Attributes
    ----------
    width : int
        If not None, the SIMD lane/SIMT block width.  Cannot be specified along with depth
    depth : int
        If not None, the SIMD lane/SIMT block depth.  Cannot be specified along with width
    ilp : bool
        If True, use the ILP tag on the species loop.  Cannot be specified along with unr
    unr : int
        If not None, the unroll length to apply to the species loop. Cannot be specified along with ilp
    layout : {'cpu', 'gpu'}
        The memory layout of the arrays
    lang : {'opencl', 'c', 'cuda'}
        One of the supported languages
    """
    def __init__(self, width=None, depth=None, ilp=False,
                    unr=None, order='cpu', lang='opencl'):
        self.width = width
        self.depth = depth
        self.ilp = ilp
        self.unr = unr
        self.order = order
        check_lang(lang)
        self.lang = lang


def get_context(device='0'):
    """
    Simple method to generate a pyopencl context

    Parameters
    ----------
    device : str
        The pyopencl string denoting the device to use, defaults to '0'
    """
    os.environ['PYOPENCL_CTX'] = device
    #os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

    lp.set_caching_enabled(False)
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    return ctx, queue

def get_header(knl):
    code, _ = lp.generate_code(knl)
    header = next(line for line in code.split('\n') if
        re.search(r'(?:__kernel(__)?)?\s*void', line))
    return header

def get_code(knl):
    code, _ = lp.generate_code(knl)
    return code

def auto_run(knl, ref_answer, device='0', **input_args):
    """
    This method tests the supplied `loopy.kernel` against a reference answer
    and a reference `loopy.kernel` if supplied

    Parameters
    ----------
    knl : `loopy.kernel`
        The kernel to test
    ref_answer : `numpy.array`
        The numpy array to test against, should be the same shape as the kernel output
    device : str
        The pyopencl string denoting the device to use, defaults to '0'
    input_args : dict of `numpy.array`s
        The arguements to supply to the kernel

    Returns
    -------
    result : bool
        True if all tests pass
    """

    #create context
    ctx, queue = get_context(device)

    #run kernel
    evt, (out,) = knl(queue, **input_args)

    #check against supplied answer
    return np.allclose(out, ref_answer)

