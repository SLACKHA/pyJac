# Python 2 compatibility
from __future__ import division
from __future__ import print_function

# Standard libraries
import os
import re

# Non-standard libraries
import sympy as sp

#local includes
from ..core.reaction_types import reaction_type, thd_body_type, falloff_form, reversible_type

#class recognition for sympy import
from . import sympy_addons as sp_add
from . import custom_sympy_classes as sp_cust
local_dict = vars(sp_cust)
from sympy.tensor.indexed import Idx, Indexed, IndexedBase
local_dict['Idx'] = Idx
local_dict['MyIndexedFuncValue'] = sp_cust.MyIndexedFunc.MyIndexedFuncValue
local_dict['IndexedFuncValue'] = sp_add.IndexedFunc.IndexedFuncValue

enum_map = [reaction_type, thd_body_type, falloff_form, reversible_type]
enum_map = {str(m.__name__) : m for m in enum_map}

def enum_from_str(condition):
    enum_class, enum_name = condition.split('.')
    if enum_class not in enum_map:
        raise Exception('Enum {} not found in reaction_types.'.format(enum_class))

    return enum_map[enum_class][enum_name]



def load_equations(conp=True, check=False):
    """
    Returns equation/symbol lists from pre-derived sympy output

    Parameters
    ----------
    conp : bool
        If true, the constant pressure derivation will be used
        If false, the constant volume derivation will be used
    check : bool
        If true, the equations input will be checked to ensure that
        all variables within are defined (slower, used for testing)

    Returns
    -------
    var_list : list of `sympy.Symbol`
        The list of variables defined in the derivation file
    eqn_list : dict of `sympy.Symbol`
        A dictionary with equation mappings for the variables in `var_list`
        If the variable is conditionally defined, eqn_list[variable] will be a sub-dictionary
        that contains definitions for sets of conditions of type `reaction_types`
    """

    script_dir = os.path.abspath(os.path.dirname(__file__))

    var_list = []
    eqn_list = {}
    with open(os.path.join(script_dir, 'con{}_derivation.sympy'.format(
        'p' if conp else 'v')), 'r') as file:
        lines = [line.strip() for line in file.readlines()]

    #first we read the variables
    for i, line in enumerate(lines):
        if not line: #reached the end of the var list
            break
        var_list.append(sp.sympify(lines[i], locals=local_dict))

    #increment to next good line
    i += 1

    #now we must parse the equations
    while i < len(lines):
        try:
            sym = sp.sympify(lines[i], locals=local_dict)
        except Exception as e:
            raise Exception('Error parsing at line: {}\n{}'.format(i, lines[i]))

        if sym in eqn_list:
            raise Exception('Sympy parsing error at line {},'.format(i) + 
                'duplicate definition found')

        #bump to definition
        i += 1
        
        #the next line is either an if statement
        conditional = None
        while True:
            if i >= len(lines) or not lines[i]:
                break
            match = re.search('^if (.+)$', lines[i])
            if match:
                #check 
                if conditional == False:
                    raise Exception('Sympy parsing error at line {},'.format(i) + 
                        'conditional definition found in non-conditional block')
                #map conditionals to enums
                conditional = True
                conditions = match.group(1).split(',')
                enums = tuple(enum_from_str(condition) for condition in conditions)

                #place sympified equation in eqn_list
                if sym not in eqn_list:
                    eqn_list[sym] = {}
                try:
                    eqn_list[sym][enums] = sp.sympify(lines[i + 1], locals=local_dict)
                except Exception as e:
                    raise Exception('Error parsing at line: {}\n{}'.format(i, lines[i]))

                i += 2
            else:
                #check 
                if conditional:
                    raise Exception('Sympy parsing error at line {},'.format(i) + 
                        'non-conditional definition found in conditional block')
                conditional = False
                #place sympified equation in eqn_list
                try:
                    eqn_list[sym] = sp.sympify(lines[i], locals=local_dict)
                except Exception as e:
                    raise Exception('Error parsing at line: {}\n{}'.format(i, lines[i]))
                i += 1
        i += 1

    #populate variable list
    var_list = set()
    time = sp.Symbol('t')
    conc = sp_cust.IndexedConc('[C]', time)
    blacklist = set([
        sp.Symbol('P'), sp.Symbol('V'),
        sp_cust.MyImplicitSymbol('P', time), sp_cust.MyImplicitSymbol('V', time),
        sp.Symbol('R_u'), sp.Symbol('P_{atm}'),
        time,
        sp_cust.MyImplicitSymbol('T', time),
        conc, conc[sp.Idx('k')], conc[sp.Idx('j')], conc[sp.Idx('m')]])
    for x in eqn_list:
        var_list = var_list.union([y for y in x.free_symbols if not (isinstance(y, Idx))])
    var_list = var_list - blacklist

    var_list = list(var_list)

    if check:
        try:
            assert all(x in eqn_list for x in var_list)
        except AssertionError as e:
            missing = next(x for x in var_list if x not in eqn_list)
            raise Exception('On check, missing equation for variable {}.'.format(missing))

    return var_list, eqn_list