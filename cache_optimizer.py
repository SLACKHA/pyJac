""" Cache Optimizer

    Reorders loads of rate and species subs to optimize cache hits, etc.
"""

# Python 2 compatibility
from __future__ import division
from __future__ import print_function
from CUDAParams import Jacob_Unroll, ResetOnJacUnroll
from pyJac import C_Jacob_Unroll
import multiprocessing
import pickle


def get_nu(species, rxn):
    if species.name in rxn.prod and species.name in rxn.reac:
        nu = (rxn.prod_nu[rxn.prod.index(species.name)] -
              rxn.reac_nu[rxn.reac.index(species.name)])
    elif species.name in rxn.prod:
        nu = rxn.prod_nu[rxn.prod.index(species.name)]
    elif species.name in rxn.reac:
        nu = -rxn.reac_nu[rxn.reac.index(species.name)]
    else:
        # doesn't participate in reaction
        return None
    return nu


def get_nu_name(sp_name, rxn):
    if sp_name in rxn.prod and sp_name in rxn.reac:
        nu = (rxn.prod_nu[rxn.prod.index(sp_name)] -
              rxn.reac_nu[rxn.reac.index(sp_name)])
    elif sp_name in rxn.prod:
        nu = rxn.prod_nu[rxn.prod.index(sp_name)]
    elif sp_name in rxn.reac:
        nu = -rxn.reac_nu[rxn.reac.index(sp_name)]
    else:
        # doesn't participate in reaction
        return None
    return nu


def get_mappings(specs, reacs, load_non_participating=False, consider_thd=False):
    r_to_s = [set() for i in range(len(reacs))]
    s_to_r = [set() for i in range(len(specs))]
    for sind, sp in enumerate(specs):
        for rind, rxn in enumerate(reacs):
            the_set = set(rxn.reac + rxn.prod)
            if consider_thd:
                thd_sp = [thd[0] for thd in rxn.thd_body_eff if thd[1] - 1.0 != 0]
                if thd_sp:
                    the_set = the_set.union(set(thd_sp))
            if any(sp.name == x for x in the_set):
                nu = get_nu(sp, rxn)
                if (nu is not None and nu != 0) or load_non_participating:
                    r_to_s[rind].add(sind)
                    s_to_r[sind].add(rind)

    return r_to_s, s_to_r


def __greedy_loop(seed, selection_pool, score_fn, additional_args, size=None, multi_thread=1):
    """
    The work horse of the greedy_optimizer

    Notes
    -----
    This method will start a list with the seed object, and greedily select the best object from the selection_pool (
    based on the score fn) until the list reaches the supplied size

    Parameters
    ----------
    seed : object
        The starting object
    selection_pool : list of object
        The pool to select from
    score_fn : function(list, object, additional_args) -> float
        Given the current list, and a potential object, this returns a score estimating how good of a choice this
        object is
        A large score should indicate a better choice
    additional_args : object
        Additional arguements to pass to the score_fn, can be None
    updater : function(obj)
        Given the selected obj, this will perform any updates needed for the score_fn
    size : int, optional
        If supplied, the returned list will be of this size, else the entire selection_pool will be used
    """

    Pool = multiprocessing.Pool(multi_thread)
    size = min(size, len(selection_pool)) if size is not None else len(selection_pool)
    the_list = [seed]
    size_offset = 1
    if seed in selection_pool:
        selection_pool.remove(seed)
        size_offset = 0

    while len(the_list) < size + size_offset:
        max_score = None
        best_candidate = None
        score_list = []
        for i, obj in enumerate(selection_pool):
            if obj in the_list:
                continue
            score_list.append((i, Pool.apply_async(score_fn, args=(the_list, obj, additional_args))))
        score_list = [(score[0], score[1].get()) for score in score_list]
        for score in score_list:
            if max_score is None or score[1] > max_score:
                max_score = score[1]
                best_candidate = selection_pool[score[0]]
        the_list.append(best_candidate)
        selection_pool.remove(best_candidate)
        # updater(best_candidate)
    return the_list

    # a score function that compares the percentage of shared species between reactions


def __shared_specs_score(the_list, candidate, additional_args):
    reacs = additional_args['reacs']
    r_to_s = additional_args['r_to_s']
    ind_1 = reacs.index(the_list[-1])
    ind_2 = reacs.index(candidate)
    return len(r_to_s[ind_1].intersection(r_to_s[ind_2])) / len(r_to_s[ind_1])


def __conc_temp(rxn, specs):
    ret_list = []
    for thd_sp in rxn.thd_body_eff:
        isp = specs.index(next((s for s in specs
                                if s.name == thd_sp[0]), None))
        val = thd_sp[1] - 1.0
        if val != 0.0:
            ret_list.append(isp)
    return ret_list

    # a score function that compares the percentage of shared species between


def __shared_specs_score_pdep(the_list, candidate, additional_args):
    reacs = additional_args['reacs']
    ind_1 = reacs.index(the_list[-1])
    ind_2 = reacs.index(candidate)
    misses = 0
    for spec, val in reacs[ind_1].thd_body_eff:
        other_val = next((sp[1] for sp in reacs[ind_2].thd_body_eff if sp[0] == spec), None)
        if other_val is None or other_val != val:
            misses += 1
    for spec, val in reacs[ind_2].thd_body_eff:
        if not any(sp[0] == spec for sp in reacs[ind_1].thd_body_eff):
            misses += 1
    for sp in set(reacs[ind_1].reac + reacs[ind_1].prod):
        nu_1 = get_nu_name(sp, reacs[ind_1])
        nu_2 = get_nu_name(sp, reacs[ind_2])
        if nu_1 is not None and nu_1 != 0 and (nu_2 is None or nu_2 == 0):
            misses += 1
    for sp in set(reacs[ind_2].reac + reacs[ind_2].prod):
        nu_1 = get_nu_name(sp, reacs[ind_1])
        nu_2 = get_nu_name(sp, reacs[ind_2])
        if nu_2 is not None and nu_2 != 0 and (nu_1 is None or nu_1 == 0):
            misses += 1

    return (1.0 - misses / len(set(reacs[ind_1].reac + reacs[ind_1].prod +
            reacs[ind_2].reac + reacs[ind_2].prod +
            [thd[0] for thd in reacs[ind_1].thd_body_eff] +
            [thd[0] for thd in reacs[ind_2].thd_body_eff]))
            )


def __get_positioning_score(reac_index, candidate_index, rxn_rate_order, r_to_s):
    # candidate is a potential list index for the reaction 'reac_index'
    # test between candidate and candidate + 1
    ind_1 = rxn_rate_order[candidate_index]
    val_1 = len(r_to_s[ind_1].intersection(r_to_s[reac_index])) / len(r_to_s[ind_1])
    if candidate_index + 1 < len(rxn_rate_order):
        ind_2 = rxn_rate_order[candidate_index + 1]
        val_2 = len(r_to_s[ind_2].intersection(r_to_s[reac_index])) / len(r_to_s[ind_2])
        val_1 = (val_1 + val_2) / 2.0
    return val_1


def __get_max_rxn(the_list, reacs, spec_scores, reac_scores):
    reac_inds = [reacs.index(reac) for reac in the_list]
    max_score = None
    max_rxn = None
    for rxn in reac_inds:
        the_count = 0
        for spec in reac_scores[rxn]:
            the_count += len(spec_scores[spec].intersection(reac_inds))
        if max_score is None or the_count > max_score:
            max_score = the_count
            max_rxn = rxn
    return reacs[max_rxn]


def greedy_optimizer(lang, specs, reacs, multi_thread, force_optimize, build_path):
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

    unroll_len = C_Jacob_Unroll if lang == 'c' else Jacob_Unroll
    # first try to load past data
    if not force_optimize:
        try:
            same_mech = False
            with open(build_path + 'optimized.pickle', 'rb') as file:
                old_specs = pickle.load(file)
                old_reacs = pickle.load(file)
                rxn_rate_order = pickle.load(file)
                spec_rate_order = pickle.load(file)
                pdep_rate_order = pickle.load(file)
                spec_ordering = pickle.load(file)
                rxn_ordering = pickle.load(file)
            same_mech = all(any(s == sp for sp in specs) for s in old_specs) and \
                        len(specs) == len(old_specs) and \
                        all(any(r == rxn for rxn in reacs) for r in old_reacs) and \
                        len(reacs) == len(old_reacs)
        except Exception, e:
            same_mech = False
        if same_mech:
            # we have to do the spec_rate_order each time
            return splittings, old_specs, old_reacs, rxn_rate_order, pdep_rate_order, spec_rate_order, spec_ordering,\
                   rxn_ordering

    splittings = []

    # First find pdep reacs
    pdep_reacs = []
    for reac in reacs:
        if reac.thd_body or reac.pdep:
            # add reaction index to list
            pdep_reacs.append(reac)

    rxn_ordering = []
    r_to_s, s_to_r = get_mappings(specs, reacs)
    args = {'r_to_s': r_to_s, 'reacs': reacs}
    reac_list = reacs[:]
    # and finally order the rest of the reactions
    while len(reac_list):
        if len(rxn_ordering) and not ResetOnJacUnroll:
            max_rxn = reacs[rxn_ordering[-1]]
        else:
            max_rxn = __get_max_rxn(reac_list, reacs, s_to_r, r_to_s)
        order = __greedy_loop(max_rxn, reac_list, __shared_specs_score, additional_args=args, size=unroll_len,
                              multi_thread=multi_thread)
        # remove the seed, it was already added
        if len(rxn_ordering) and not ResetOnJacUnroll:
            order = order[1:]
        rxn_ordering.extend([reacs.index(reac) for reac in order])
        reac_list = [reac for reac in reac_list if not reac in order]
        splittings.append(len(order))

    # the reactions order is now determined, so let's reorder them
    temp = reacs[:]
    reacs = [temp[i] for i in rxn_ordering]

    # up next we will reorder the species so they are stored in the order that we will be loading them in the jacobian
    spec_ordering = []
    for reac in reacs:
        for spec_name in set(reac.reac + reac.prod):
            spec = next(sp for sp in specs if sp.name == spec_name)
            if get_nu(spec, reac):
                isp = specs.index(spec)
                if isp not in spec_ordering:
                    spec_ordering.append(isp)

    # make sure we have them all (i.e. 3rd body only)
    for isp in range(len(specs)):
        if not isp in spec_ordering:
            spec_ordering.append(isp)

    # now reorder
    temp = specs[:]
    specs = [temp[i] for i in spec_ordering]

    # we have to update our mappings now that things are reordered
    r_to_s, s_to_r = get_mappings(specs, reacs)

    rxn_rate_order = range(len(reacs))
    pdep_rate_order = [reacs.index(reac) for reac in pdep_reacs]
    spec_rate_order = []
    for i in range(len(reacs)):
        spec_rate_order.append((list(sorted(r_to_s[i])), [i]))

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
        pickle.dump(rxn_rate_order, file)
        pickle.dump(spec_rate_order, file)
        pickle.dump(pdep_rate_order, file)
        pickle.dump(print_spec_order, file)
        pickle.dump(print_rxn_order, file)

    # complete, so now return
    return specs, reacs, rxn_rate_order, pdep_rate_order, spec_rate_order, print_spec_order, print_rxn_order
