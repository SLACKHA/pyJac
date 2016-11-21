from __future__ import print_function

#package imports
from enum import Enum
import loopy as lp
import numpy as np
import pyopencl as cl
import re
import os

#local imports
from ..utils import check_lang

class RateSpecialization(Enum):
    fixed = 0,
    hybrid = 1,
    full = 2


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
    order : {'C', 'F'}
        The memory layout of the arrays, C (row major) or Fortran (column major)
    lang : {'opencl', 'c', 'cuda'}
        One of the supported languages
    rate_spec : RateSpecialization
        Controls the level to which Arrenhius rate evaluations are specialized
    rate_spec_kernels : bool
        If True, break different Arrenhius rate specializations into different kernels

    """
    def __init__(self, width=None, depth=None, ilp=False,
                    unr=None, order='cpu', lang='opencl',
                    rate_spec=RateSpecialization.fixed,
                    rate_spec_kernels=False):
        self.width = width
        self.depth = depth
        self.ilp = ilp
        self.unr = unr
        self.order = order
        check_lang(lang)
        self.lang = lang
        self.rate_spec = rate_spec
        self.rate_spec_kernels = rate_spec_kernels


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
    """
    Returns header definition code for a `loopy.kernel`

    Parameters
    ----------
    knl : `loopy.kernel`
        The kernel to generate a header definition for

    Returns
    -------
    Generated device header code

    Notes
    -----
    The kernel's Target and name should be set for proper functioning
    """
    code, _ = lp.generate_code(knl)
    header = next(line for line in code.split('\n') if
        re.search(r'(?:__kernel(__)?)?\s*void', line))
    return header

def get_code(knl):
    """
    Returns the device code for a `loopy.kernel`

    Parameters
    ----------
    knl : `loopy.kernel`
        The kernel to generate code for

    Returns
    -------
    Generated device code

    Notes
    -----
    The kernel's Target and name should be set for proper functioning
    """
    code, _ = lp.generate_code(knl)
    return code

def auto_run(knl, ref_answer, compare_mask=None, compare_axis=0, device='0', **input_args):
    """
    This method tests the supplied `loopy.kernel` (or list thereof) against a reference answer

    Parameters
    ----------
    knl : `loopy.kernel` or list of `loopy.kernel`
        The kernel to test, if a list of kernels they will be successively applied and the
        end result compared
    ref_answer : `numpy.array`
        The numpy array to test against, should be the same shape as the kernel output
    compare_mask : `numpy.array`
        A list of indexes to compare, useful when the kernel only computes partial results
    compare_axis = int
        An axis to apply the compare_mask along, unused if compare_mask is none
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
    if isinstance(knl, list):
        out_ref = np.zeros_like(ref_answer)
        for k in knl:
            try:
                evt, (out,) = k(queue, **input_args)
            except Exception as e:
                print(k)
                raise e
            copy_inds = np.where(np.logical_not(np.isinf(out)))
            out_ref[copy_inds] = out[copy_inds]
        out = out_ref
    else:
        try:
            evt, (out,) = knl(queue, **input_args)
        except Exception as e:
            print(knl)
            raise e

    if compare_mask:
        return np.allclose(np.take(out, compare_mask, compare_axis),
            np.take(ref_answer, compare_mask, compare_axis))
    #check against supplied answer
    return np.allclose(out, ref_answer)
