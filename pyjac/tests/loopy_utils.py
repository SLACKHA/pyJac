import loopy as lp
import numpy as np
import pyopencl as cl

import os

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

