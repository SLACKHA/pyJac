""" Cache Optimizer

    Reorders loads of rate and species subs to optimize cache hits, etc.
"""

# Python 2 compatibility
from __future__ import division
from __future__ import print_function

# Standard libraries
import multiprocessing
import pickle
import os

# Local imports
from CUDAParams import Jacob_Unroll, ResetOnJacUnroll
from CParams import C_Jacob_Unroll
import utils

try:
    from Numberjack import *
    Model().load('SCIP')
    HAVE_NJ = True
except Exception, e:
    HAVE_NJ = False
    print(e)
    print('Cache-optimization support disabled...')
    class Variable(object):
        def __init__(self, lb, ub, name):
            pass

def get_mappings(specs, reacs, load_non_participating=False, consider_thd=False):
    r_to_s = [set() for i in range(len(reacs))]
    s_to_r = [set() for i in range(len(specs))]
    for sind, sp in enumerate(specs):
        for rind, rxn in enumerate(reacs):
            the_set = set(rxn.reac + rxn.prod)
            if consider_thd:
                thd_sp = [thd[0] for thd in rxn.thd_body_eff if thd[1] != 0]
                if thd_sp:
                    the_set = the_set.union(set(thd_sp))
            if any(sp.name == x for x in the_set):
                nu = utils.get_nu(sp, rxn)
                if (nu is not None and nu != 0) or load_non_participating:
                    r_to_s[rind].add(sind)
                    s_to_r[sind].add(rind)

    return r_to_s, s_to_r

class rxn_comp(Variable):
    def __init__(self, c1, c2, lb=0, ub=None):
        ub = len(c1)
        Variable.__init__(self, lb, ub, "rxn_comp")
        self.children = c1 + c2
        self.c1 = c1
        self.c2 = c2

    def decompose(self):
        '''
        Decompose must return either a list containing a list of expressions
        '''

        return [ Sum([self.c1[i] < self.c2[i] for i in range(len(self.c1))]) ]


class sp_comp(Variable):
    def __init__(self, r1, r2, lb=0, ub=None):
        ub = len(r1)
        Variable.__init__(self, lb, ub, "rxn_comp")
        self.children = r1 + r2
        self.r1 = r1
        self.r2 = r2

    def decompose(self):
        '''
        Decompose must return either a list containing a list of expressions
        '''

        return [ Sum([self.r2[i] < self.r1[i] for i in range(len(self.r1))]) ]


def optimize_cache(specs, reacs, multi_thread,
                    force_optimize, build_path,
                     last_spec, time_lim=60, verbosity=1):
    """
    Utilizes the Numberjack package to optimize species
    and reaction orders in the mechanism to attempt to improve cache hit rates.

    Parameters
    ----------
    specs : list of SpecInfo
        List of species in the mechanism.
    reacs : list of ReacInfo
        List of reactions in the mechanism.
    multi_thread : int
        The number of threads to use during optimization
    force_optimize : bool
        If true, reoptimize even if past data is available
    build_path : str
        The path to the build directory
    last_spec : int
        The index of the species that should be placed last
    time_lim : int
        The time limit for optimization operations in minutes
    verbosity : int
        The verbosity of the Numberjack solvers

    Returns
    _______

    specs : list of SpecInfo
        The reordered list of species in the mechanism

    reacs : list of ReacInfo
        the reordered list of reacs in the mechanism

    fwd_spec_mapping : list of int
        A mapping of the original mechanism to the new species order

    fwd_rxn_mapping : list of int
        A mapping of the original mechanism to the new reaction order

    reverse_spec_mapping : list of int
        A mapping of the new species order to the original mechanism

    reverse_rxn_mapping : list of int
        A mapping of the new reaction order to the original mechanism
    """

    print('Beginning Cache-optimization process...')
    # first try to load past data
    if not force_optimize:
        print('Checking for old optimization')
        try:
            same_mech = False
            with open(os.path.join(build_path, 'optimized.pickle'), 'rb') as file:
                old_specs = pickle.load(file)
                old_reacs = pickle.load(file)
                fwd_spec_mapping = pickle.load(file)
                fwd_rxn_mapping = pickle.load(file)
                reverse_spec_mapping = pickle.load(file)
                reverse_rxn_mapping = pickle.load(file)
            same_mech = all(any(s == sp for sp in specs) for s in old_specs) and \
                        len(specs) == len(old_specs) and \
                        all(any(r == rxn for rxn in reacs) for r in old_reacs) and \
                        len(reacs) == len(old_reacs)
            if fwd_spec_mapping[last_spec] != len(specs) - 1:
                print('Different last species detected, old species was {} and new species is {}'.format(
                    specs[fwd_spec_mapping[-1]].name, specs[last_spec].name))
                print('Forcing reoptimization...')
                same_mech = False

        except Exception, e:
            print('Old optimization file not found, or does not match current mechanism... forcing optimization')
            same_mech = False
        if same_mech:
            print('Old optimization file matching current mechanism found... returning previous optimization')
            # we have to do the spec_rate_order each time
            return old_specs, old_reacs, fwd_spec_mapping, fwd_rxn_mapping, \
                    reverse_spec_mapping, reverse_rxn_mapping

    #otherwise swap the last_spec
    temp = specs[last_spec]
    specs[last_spec] = specs[-1]
    specs[-1] = temp

    print('Beginning Reaction Cache Locality Reordering')
    #get the mappings
    r_to_s, s_to_r = get_mappings(specs, reacs, consider_thd=False, load_non_participating=True)

    #set up the Numberjack constraint problem to optimize reaction order
    model = Model()

    rxn_order = [Variable(0, len(reacs), "rxn_{}".format(i)) for i in range(len(reacs))]

    #set up the constraints to ensure unique rxn placement
    model.add(AllDiff(rxn_order))

    #now set up score function
    score_matrix = Matrix(len(specs), len(reacs), 2)

    #set up the r_to_s constraints
    for i, rxn in enumerate(rxn_order):
        for sp in range(len(specs)):
            val = 1 if sp in r_to_s[i] else 0
            model.add(score_matrix[sp, rxn] == val)

    score_list = []
    for i in range(len(reacs) - 1):
        score_list.append(rxn_comp(score_matrix.col[i], score_matrix.col[i + 1]))

    score = Sum(score_list)
    model.add(Minimize(score))
    solver = model.load('SCIP')
    solver.setThreadCount(multi_thread)
    solver.setTimeLimit(time_lim * 60)
    solver.setVerbosity(verbosity)

    def __rxn_score_check(r_to_s, ordering):
        the_sum = 0
        for i in range(len(r_to_s) - 1):
            the_sum += len(r_to_s[ordering[i + 1]].difference(r_to_s[ordering[i]]))

        assert len(set(ordering)) == len(ordering)
        return the_sum

    solved = solver.solve()
    solns = [x for x in solver.solutions()]
    print("Solution Found: {}\nSolution Optimal: {}\n".format(solved, solver.is_opt()))
    if solved and solver.is_opt():
        print('Checking optimal solution')
        ordering = [x.get_value() for x in rxn_order]
    elif solved:
        print('Checking other {} solutions'.format(len(solns)))
        bestVal = None
        for solution in solns:
            val = __rxn_score_check(r_to_s, solution)
            if bestVal is None or val < bestVal:
                ordering[:] = solution
                bestVal = val
    else:
        raise Exception('No solution found, try a longer timelimit')

    pre = __rxn_score_check(r_to_s, range(len(reacs)))
    post = __rxn_score_check(r_to_s, ordering)
    print('Reaction Cache Locality Heuristic changed from {} to {}'.format(pre, post))

    if post >= pre:
        print('Using newly optimized reaction order:\n' + ', '.join(
                [str(x) for x in enumerate(ordering)]))
        fwd_rxn_mapping = ordering[:]
    else:
        print('Using original reaction order:')
        fwd_rxn_mapping = range(len(reacs))


    print('Beginning Species Cache Locality reordering')
    #now set up the species optimization problem
    #set up the Numberjack constraint problem to optimize reaction order
    model = Model()

    sp_order = [Variable(0, len(specs) - 1, "sp_{}".format(i)) for i in range(len(specs) - 1)]

    #set up the constraints to ensure unique rxn placement
    model.add(AllDiff(sp_order))

    #now set up score function
    score_matrix = Matrix(len(specs) - 1, len(reacs), 2)

    #set up the r_to_s constraints
    for i, sp in enumerate(sp_order):
        for rxn in range(len(reacs)):
            val = 1 if rxn in s_to_r[i] else 0
            model.add(score_matrix[sp, rxn] == val)

    score_list = []
    for i in range(len(specs) - 2):
        score_list.append(sp_comp(score_matrix.row[i], score_matrix.row[i + 1]))

    score = Sum(score_list)
    model.add(Minimize(score))
    solver = model.load('SCIP')
    solver.setThreadCount(multi_thread)
    solver.setVerbosity(verbosity)
    solver.setTimeLimit(time_lim * 60)

    def __sp_score_check(s_to_r, ordering):
        the_sum = 0
        for i in range(len(s_to_r) - 2):
            the_sum += len(s_to_r[ordering[i + 1]].difference(s_to_r[ordering[i]]))

        assert len(set(ordering)) == len(ordering)
        return the_sum

    solved = solver.solve()
    solns = [x for x in solver.solutions()]
    print("Solution Found: {}\nSolution Optimal: {}\n".format(solved, solver.is_opt()))
    if solved and solver.is_opt():
        print('Checking optimal solution')
        ordering = [x.get_value() for x in sp_order]
    elif solved:
        print('Checking other {} solutions'.format(len(solns)))
        bestVal = None
        for solution in solns:
            val = __sp_score_check(s_to_r, solution)
            if bestVal is None or val < bestVal:
                ordering[:] = solution
                bestVal = val
    else:
        raise Exception('No solution found, try a longer timelimit')

    pre = __sp_score_check(s_to_r, range(len(specs)))
    post = __sp_score_check(s_to_r, ordering)
    print('Species Cache Locality Heuristic changed from {} to {}'.format(pre, post))

    if post >= pre:
        fwd_spec_mapping = ordering[:]
        #the last species is the last_spec
        fwd_spec_mapping.insert(last_spec, len(specs) - 1)
        print('Using newly optimized species order:\n' + ', '.join(
            [str(x) for x in enumerate(fwd_spec_mapping)]))
    else:
        print('Using original species order')
        fwd_spec_mapping = range(len(specs))
        fwd_spec_mapping[last_spec:-1] = fwd_spec_mapping[last_spec + 1:]
        fwd_spec_mapping[-1] = last_spec

    #swap the last species back, for simplicity in reordering
    temp = specs[last_spec]
    specs[last_spec] = specs[-1]
    specs[-1] = temp

    #reorder the species and reactions in the appropriate order
    spec_temp = specs[:]
    specs = [specs[i] for i in fwd_spec_mapping]
    reac_temp = reacs[:]
    reacs = [reacs[i] for i in fwd_rxn_mapping]

    reverse_spec_mapping = []
    # finally reorder the spec and rxn orderings to fix for printing
    for spec_ind in range(len(fwd_spec_mapping)):
        reverse_spec_mapping.append(
            fwd_spec_mapping.index(spec_ind)
        )

    reverse_rxn_mapping = []
    for rxn_ind in range(len(fwd_rxn_mapping)):
        reverse_rxn_mapping.append(
            fwd_rxn_mapping.index(rxn_ind)
        )

    # save to avoid reoptimization if possible
    with open(os.path.join(build_path, 'optimized.pickle'), 'wb') as file:
        pickle.dump(specs, file)
        pickle.dump(reacs, file)
        pickle.dump(fwd_spec_mapping, file)
        pickle.dump(fwd_rxn_mapping, file)
        pickle.dump(reverse_spec_mapping, file)
        pickle.dump(reverse_rxn_mapping, file)

    # complete, so now return
    return specs, reacs, fwd_spec_mapping, fwd_rxn_mapping, reverse_spec_mapping, reverse_rxn_mapping
