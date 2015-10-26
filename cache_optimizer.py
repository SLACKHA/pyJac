""" Cache Optimizer

    Reorders loads of rate and species subs to optimize cache hits, etc.
"""

# Python 2 compatibility
from __future__ import division
from __future__ import print_function
from CUDAParams import Jacob_Unroll, ResetOnJacUnroll
from CParams import C_Jacob_Unroll
import utils
import multiprocessing
import pickle
import timeit

try:
    from Numberjack import *
    Model().load('SCIP')
    HAVE_NJ = True
except Exception, e:
    HAVE_NJ = False
    print(e)
    print('Cache-optimization support disabled...')

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

        return [ Sum([self.r1[i] < self.r2[i] for i in range(len(self.r1))]) ]


def greedy_optimizer(lang, specs, reacs, multi_thread, force_optimize, build_path, time_lim=60, verbosity=1):
    """
    An optimization method that reorders the species and reactions in a method to attempt to keep data in cache as
    long as possible

    Notes
    -----
    This method optimizes based on Jacobian matrix generation, as this is the most important and time consuming step
    of the reaction rate subroutines
    Species and reactions are reordered to optimize the cache rates for this.

    Next orderings for the evaluation of reactions, pressure dependent reactions and species rates are determined in
    order to optimize the various rate routines

    Parameters
    ----------
    lang : str
        The language
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
    time_lim : int
        The time limit for optimization operations in minutes

    Returns
    _______

    specs : list of SpecInfo
        The reordered list of species in the mechanism

    reacs : list of ReacInfo
        the reordered list of reacs in the mechanism

    rxn_rate_order : list of int
        A list indicating the order to evaluate reaction rates in rxn_rates subroutine

    pdep_rate_order : list of int
        A list indicating the order to evaluate pressure dependent rates in rxn_rates_pres_mod subroutine

    spec_rate_order : list of int
        A list indicated the order to evaluate species rates in spec_rates subroutine

    old_species_order : list of int
        A list indicating the positioning of the species in the original mechanism, used in rate testing

     old_rxn_order : list of int
        A list indicating the positioning of the reactions in the original mechanism, used in rate testing
    """

    print('Beginning Cache-optimization process...')
    if not HAVE_NJ:
        print("Cache-optimization disabled, returning original mechanism")
        spec_rate_order = range(len(specs))
        rxn_rate_order = range(len(reacs))
        if any(r.pdep or r.thd_body for r in reacs):
            pdep_rate_order = [x for x in range(len(reacs))
                               if reacs[x].pdep or reacs[x].thd_body
                               ]
        else:
            pdep_rate_order = None
        old_spec_order = range(len(specs))
        old_rxn_order = range(len(reacs))

        return specs, reacs, rxn_rate_order, spec_rate_order, pdep_rate_order, old_spec_order, \
                old_rxn_order

    unroll_len = C_Jacob_Unroll if lang == 'c' else Jacob_Unroll
    # first try to load past data
    if not force_optimize:
        print('Checking for old optimization')
        try:
            same_mech = False
            with open(build_path + 'optimized.pickle', 'rb') as file:
                old_specs = pickle.load(file)
                old_reacs = pickle.load(file)
                rxn_rate_order = pickle.load(file)
                pdep_rate_order = pickle.load(file)
                spec_rate_order = pickle.load(file)
                print_spec_ordering = pickle.load(file)
                print_rxn_ordering = pickle.load(file)
            same_mech = all(any(s == sp for sp in specs) for s in old_specs) and \
                        len(specs) == len(old_specs) and \
                        all(any(r == rxn for rxn in reacs) for r in old_reacs) and \
                        len(reacs) == len(old_reacs)
        except Exception, e:
            print('Old optimization file not found, or does not match current mechanism... forcing optimization')
            same_mech = False
        if same_mech:
            print('Old optimization file matching current mechanism found... returning previous optimization')
            # we have to do the spec_rate_order each time
            return old_specs, old_reacs, rxn_rate_order, pdep_rate_order, spec_rate_order, \
                    print_spec_ordering, print_rxn_ordering

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
        rxn_ordering = ordering[:]
    else:
        print('Using original reaction order:')
        rxn_ordering = range(len(reacs))


    print('Beginning Species Cache Locality reordering')
    #now set up the species optimization problem
    #set up the Numberjack constraint problem to optimize reaction order
    model = Model()

    sp_order = [Variable(0, len(specs), "sp_{}".format(i)) for i in range(len(specs))]

    #set up the constraints to ensure unique rxn placement
    model.add(AllDiff(sp_order))

    #now set up score function
    score_matrix = Matrix(len(specs), len(reacs), 2)

    #set up the r_to_s constraints
    for i, sp in enumerate(sp_order):
        for rxn in range(len(reacs)):
            val = 1 if rxn in s_to_r[i] else 0
            model.add(score_matrix[sp, rxn] == val)

    score_list = []
    for i in range(len(specs) - 1):
        score_list.append(sp_comp(score_matrix.row[i], score_matrix.row[i + 1]))

    score = Sum(score_list)
    model.add(Minimize(score))
    solver = model.load('SCIP')
    solver.setThreadCount(multi_thread)
    solver.setVerbosity(verbosity)
    solver.setTimeLimit(time_lim * 60)

    def __sp_score_check(s_to_r, ordering):
        the_sum = 0
        for i in range(len(s_to_r) - 1):
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
        print('Using newly optimized species order:\n' + ', '.join(
            [str(x) for x in enumerate(ordering)]))
        spec_ordering = ordering[:]
    else:
        print('Using original species order')
        spec_ordering = range(len(specs))

    pdep_rate_order = [i for i, rxn in enumerate(reacs) if rxn.pdep or rxn.thd_body]
    pdep_rate_order = [rxn_ordering.index(i) for i in pdep_rate_order]
    print_spec_order = []
    # finally reorder the spec and rxn orderings to fix for printing
    for spec_ind in range(len(spec_ordering)):
        print_spec_order.append(
            spec_ordering.index(spec_ind)
        )

    print_rxn_order = []
    for rxn_ind in range(len(rxn_ordering)):
        print_rxn_order.append(
            rxn_ordering.index(rxn_ind)
        )

    # save to avoid reoptimization if possible
    with open(build_path + 'optimized.pickle', 'wb') as file:
        pickle.dump(specs, file)
        pickle.dump(reacs, file)
        pickle.dump(rxn_ordering, file)
        pickle.dump(pdep_rate_order, file)
        pickle.dump(spec_ordering, file)
        pickle.dump(print_spec_order, file)
        pickle.dump(print_rxn_order, file)

    # complete, so now return
    return specs, reacs, rxn_ordering, pdep_rate_order, spec_ordering, print_spec_order, print_rxn_order
