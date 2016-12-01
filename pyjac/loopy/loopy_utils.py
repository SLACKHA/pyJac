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
from .loopy_edit_script import substitute as codefix

edit_script = os.path.join(os.path.abspath(os.path.dirname(__file__)),
    'loopy_edit_script.py')

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
    os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

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

def set_editor(knl):
    """
    Returns a copy of knl set up for various automated bug-fixes

    Parameters
    ----------
    knl : `loopy.kernel`
        The kernel to generate code for

    Returns
    -------
    edit_knl : `loopy.kernel`
        The kernel set up for editing
    """

    #set the edit script as the 'editor'
    if not 'EDITOR' in os.environ:
        os.environ['EDITOR'] = edit_script

    #turn on code editing
    edit_knl = lp.set_options(knl, edit_code=True)

    return edit_knl

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
    return codefix('stdin', text_in=code)

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
            test_knl = set_editor(k)
            try:
                evt, (out,) = test_knl(queue, **input_args)
            except Exception as e:
                print(test_knl)
                raise e
            copy_inds = np.where(np.logical_not(np.isinf(out)))
            out_ref[copy_inds] = out[copy_inds]
        out = out_ref
    else:
        try:
            test_knl = set_editor(knl)
            evt, (out,) = test_knl(queue, **input_args)
        except Exception as e:
            print(test_knl)
            raise e

    if compare_mask:
        return np.allclose(np.take(out, compare_mask, compare_axis),
            np.take(ref_answer, compare_mask, compare_axis))
    #check against supplied answer
    return np.allclose(out, ref_answer)

def generate_map_instruction(oldname, newname, map_arr):
    """
    Generates a loopy instruction that maps oldname -> newname via the
    mapping array

    Parameters
    ----------
    oldname : str
        The old index to map from
    newname : str
        The new temporary variable to map to
    map_arr : str
        The array that holds the mappings

    Returns
    -------
    map_inst : str
        A strings to be used `loopy.Instruction`'s) for
                given mapping
    """

    return  '<>{newname} = {mapper}[{oldname}]'.format(
            newname=newname,
            mapper=map_arr,
            oldname=oldname)



def get_loopy_arg(arg_name, indicies, dimensions,
                    order='F', map_name=None):
    """
    Convience method that generates a loopy GlobalArg with correct indicies
    and sizes.

    Parameters
    ----------
    arg_name : str
        The name of the array
    indicies : list of str
        The `loopy.inames` in :param:`default_order`
    dimensions : list of str/int
        The dimensions of the `loopy.inames` in :param:`default_order`
    order : {'C', 'F'}
        The memory layout of the arrays, C (row major) or Fortran
        (column major)
    map_name : dict
        If not None, contains replacements for various indicies

    Returns
    -------
    arg_dict : dict
        A dictionary with the following keys:
            * arg : `loopy.GlobalArg`
                The generated loopy arg
            * arg_str : str
                A string form of the argument
            * map_instructs : list of str
                A list of strings to be used `loopy.Instruction`'s for
                given mappings
    """

    string_inds = indicies[:]
    map_instructs = {}
    for imap in map_name:
        #make a new name off the replaced iname
        mapped_name = '{}_map'.format(imap)
        if map_name[imap].startswith('<>'):
            #already an instruction
            map_instructs[imap] = map_name[imap]
            continue
        #add a mapping instruction
        map_instructs[imap] = generate_map_instruction(
                                            newname=mapped_name,
                                            map_arr=map_name[imap],
                                            oldname=imap)
        #and replace the index
        string_inds[string_inds.index(imap)] = mapped_name

    #the ordering / indexing of the array depends on the memory layout
    #if it's row-major, we must reverse the indicies / dimensions
    #as we assume column-major in pyJac
    if order == 'C':
        string_inds = string_inds[::-1]
        dimensions = dimensions[::-1]

    #finally make the arguement
    arg = lp.GlobalArg(arg_name, shape=tuple(dimensions), dtype=np.float64)

    #and return
    return {'arg' : arg,
            'arg_str' : '{name}[{inds}]'.format(name=arg_name,
                inds=','.join(string_inds)),
            'map_instructs' : map_instructs}

def get_target(lang):
    """

    Parameters
    ----------
    lang : str
        One of the supported languages, {'c', 'cuda', 'opencl'}

    Returns
    -------
    The correct loopy target type
    """

    check_lang(lang)

    #set target
    if lang == 'opencl':
        return lp.PyOpenCLTarget()
    elif lang == 'c':
        return lp.CTarget()
    elif lang == 'cuda':
        return lp.CudaTarget()
