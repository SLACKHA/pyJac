""" Cache Optimizer 

    Reorders loads of rate and species subs to optimize cache hits, etc.
"""

# Python 2 compatibility
from __future__ import division
from __future__ import print_function
from CUDAParams import Jacob_Unroll

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


def get_mappings(specs, reacs, load_non_participating=False, consider_thd=False):
    r_to_s = [set() for i in range(len(reacs))]
    s_to_r = [set() for i in range(len(specs))]
    for sind, sp in enumerate(specs):
        for rind, rxn in enumerate(reacs):
            the_set = set(rxn.reac + rxn.prod)
            if consider_thd:
                thd_sp = [thd[0] for thd in rxn.thd_body]
                if thd_sp:
                    the_set = the_set.union(set(thd_sp))
            if any(sp.name == x for x in the_set):
                nu = get_nu(sp, rxn)
                if (nu is not None and nu != 0) or load_non_participating:
                    r_to_s[rind].add(sind)
                    s_to_r[sind].add(rind)

    return r_to_s, s_to_r

def __greedy_loop(seed, selection_pool, score_fn, size=None):
    """
    The work horse of the greedy_optimizer

    Notes
    -----
    This method will start a list with the seed object, and greedily select the best object from the selection_pool (based on the score fn) until the list reaches the supplied size

    Parameters
    ----------
    seed : object
        The starting object
    selection_pool : list of object
        The pool to select from
    score_fn : function(list, object) -> float
        Given the current list, and a potential object, this returns a score estimating how good of a choice this object is
        A large score should indicate a better choice
    updater : function(obj)
        Given the selected obj, this will perform any updates needed for the score_fn
    size : int, optional
        If supplied, the returned list will be of this size, else the entire selection_pool will be used
    """

    size = min(size, len(selection_pool)) if size is not None else len(selection_pool)
    the_list = [seed]
    size_offset = 1
    if seed in selection_pool:
        selection_pool.remove(seed)
        size_offset = 0

    while len(the_list) < size + size_offset:
        max_score = None
        best_candidate = None
        for obj in selection_pool:
            if obj in the_list:
                continue
            score = score_fn(the_list, obj)
            if max_score is None or score > max_score:
                max_score = score
                best_candidate = obj
        the_list.append(best_candidate)
        selection_pool.remove(best_candidate)
        #updater(best_candidate)
    return the_list

def greedy_optimizer(lang, specs, reacs):
    """
    An optimization method that reorders the species and reactions in a method to attempt to keep data in cache as long as possible

    Notes
    -----
    This method optimizes based on Jacobian matrix generation, as this is the most important and time consuming step of the reaction rate subroutines
    Species and reactions are reordered to optimize the cache rates for this.

    Next orderings for the evaluation of reactions, pressure dependent reactions and species rates are determined in order to optimize the various rate routines

    Parameters
    ----------
    specs : list of SpecInfo
        List of species in the mechanism.
    reacs : list of ReacInfo
        List of reactions in the mechanism.

    Returns
    _______
    splittings : list of int

        The reaction splitings to be used in Jacobian creation (for CUDA) as a list
        i.e. [10, 20, 20] would correspond to 10 reactions in jacob_0.cu, 20 in jacob_1.cu...

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
    """

    splittings = []

    class shared_specs(object):
        def __init__(self, r_to_s, reacs):
            self.r_to_s = r_to_s
            self.reacs = reacs

         #a score function that compares the percentage of shared species between reactions
        def get_score(self, the_list, candidate):
            ind_1 = self.reacs.index(the_list[-1])
            ind_2 = self.reacs.index(candidate)
            return len(self.r_to_s[ind_1].intersection(self.r_to_s[ind_2])) / len(self.r_to_s[ind_1])

    def get_positioning_score(reac_index, candidate_index, rxn_rate_order, r_to_s):
        #candidate is a potential list index for the reaction 'reac_index'
        #test between candidate and candidate + 1
        ind_1 = rxn_rate_order[candidate_index]
        val_1 = len(r_to_s[ind_1].intersection(r_to_s[reac_index])) / len(r_to_s[ind_1])
        if candidate_index + 1 < len(rxn_rate_order):
            ind_2 = rxn_rate_order[candidate_index + 1]
            val_2 = len(r_to_s[ind_2].intersection(r_to_s[reac_index])) / len(r_to_s[ind_2])
            val_1 = (val_1 + val_2) / 2.0
        return val_1


    def __conc_temp(rxn, specs):
        ret_list = []
        for thd_sp in rxn.thd_body:
            isp = specs.index(next((s for s in specs
                                    if s.name == thd_sp[0]), None))
            val = thd_sp[1] - 1.0
            if val != 0.0:
                ret_list.append(isp)
        return ret_list

    #First find pdep reacs
    pdep_reacs = []
    for reac in reacs:
        if reac.thd or reac.pdep:
            # add reaction index to list
            pdep_reacs.append(reac)

    if len(pdep_reacs):
        pdep_r_to_s, pdep_s_to_r = get_mappings(specs, reacs, consider_thd=True, load_non_participating=True)

    #First get all the pdep rxn's with thd body's which require the conc_temp = (m + spec_1 + spec_2...)
    conc_reacs = []
    for reac in pdep_reacs:
        if reac.pdep and reac.thd_body:
            conc_reacs.append(reac)

    if len(conc_reacs):
        #get ordering
        max_rxn = conc_reacs[max(range(len(conc_reacs)), key=lambda i: len(pdep_r_to_s[i]))]
        scorer = shared_specs(pdep_r_to_s, reacs)
        conc_reacs = __greedy_loop(max_rxn, conc_reacs, scorer.get_score)

    r_to_s, s_to_r = get_mappings(specs, reacs)
    scorer = shared_specs(r_to_s, reacs)
    #next, get all the pdep ones that *aren't* in the conc_reacs
    other_pdep = [reac for reac in pdep_reacs if reac not in conc_reacs]
    if len(other_pdep):
        if len(conc_reacs):
            max_rxn = conc_reacs[-1]
        else:
            max_rxn = other_pdep[max(range(len(other_pdep)), key=lambda i: len(r_to_s[i]))]
        other_pdep = __greedy_loop(max_rxn, other_pdep, scorer.get_score)
        if len(conc_reacs):
            #need to remove the seed
            other_pdep = other_pdep[1:]

    rxn_ordering = [reacs.index(reac) for reac in conc_reacs] + [reacs.index(reac) for reac in other_pdep]
    if len(rxn_ordering):
        #add a split
        splittings.append(len(conc_reacs) + len(other_pdep))

    #and finally order the rest of the reactions
    other_reacs = [reac for reac in reacs if not (reac in conc_reacs or reac in other_pdep)]
    while len(other_reacs):
        if len(rxn_ordering):
            max_rxn = reacs[rxn_ordering[-1]]
        else:
            max_rxn = other_reacs[max(range(len(other_reacs)), key=lambda i: len(r_to_s[i]))]
        order = __greedy_loop(max_rxn, other_reacs, scorer.get_score, Jacob_Unroll)
        #remove the seed, it was already added
        order = order[1:]
        rxn_ordering.extend([reacs.index(reac) for reac in order])
        other_reacs = [reac for reac in other_reacs if not reac in order]
        splittings.append(len(order))

    #the reactions order is now determined, so let's reorder them
    temp = reacs[:]
    reacs = [temp[i] for i in rxn_ordering]

    #up next we will reorder the species so they are stored in the order that we will be loading them in the jacobian

    spec_ordering = []
    for reac in reacs:
        if reac.thd_body:
            thd_body = __conc_temp(reac, specs)
            spec_ordering.extend([thd for thd in thd_body if not thd in spec_ordering])
        for spec_name in set(reac.reac + reac.prod):
            spec = next(sp for sp in specs if sp.name == spec_name)
            isp = specs.index(spec)
            if isp not in spec_ordering:
                spec_ordering.append(isp)

    #now reorder
    temp = specs[:]
    specs = [temp[i] for i in spec_ordering]

    #next up, we have to determine the orderings for the various rate subs

    #rxn rates is easy, everything in other_reacs is already in order, we simply need to find a good spot for everything in conc_reacs and other_pdep
    rxn_rate_order = [reacs.index(reac) for reac in reacs if not (reac in other_pdep or reac in conc_reacs)]
    for reac in conc_reacs + other_pdep:
        reac_index = reacs.index(reac)
        max_score = None
        store_i = None
        for i in range(len(rxn_rate_order)):
            score = get_positioning_score(reac_index, i, rxn_rate_order, r_to_s)
            if max_score is None or score > max_score:
                max_score = score
                store_i = i
        rxn_rate_order.insert(store_i, reacs.index(reac))

    #next we do the pdep reactions, the conc_reacs are already in order
    pdep_rate_order = [pdep_reacs.index(reac) for reac in conc_reacs]
    skip = None
    if not len(pdep_rate_order):
        pdep_rate_order.append(max(range(len(other_pdep)), key=lambda i: len(pdep_r_to_s[i])))
        skip = reacs[pdep_rate_order[0]]

    for reac in other_pdep:
        if skip is not None and reac == skip:
            continue
        reac_index = reacs.index(reac)
        max_score = None
        store_i = None
        for i in range(len(pdep_rate_order)):
            score = get_positioning_score(reac_index, i, pdep_rate_order, r_to_s)
            if max_score is None or score > max_score:
                max_score = score
                store_i = i
        pdep_rate_order.insert(store_i, reacs.index(reac))

    spec_rate_order = []
    #species ordering is a bit trickier
    if lang == 'cuda':
        #cuda is much better with many independent statements
        #so simply iterate through reactions and add to each species
        for i in range(len(reacs)):
            spec_rate_order.append((list(r_to_s[i]), [i]))
    else:
        #otherwise, we're just going to keep it as is for the moment
        #on the CPU the memory latency shouldn't particularly be an issue
        for i in range(len(specs)):
            spec_rate_order.append(([i], list(s_to_r[i])))

    #complete, so now return
    return splittings, specs, reacs, rxn_rate_order, pdep_rate_order, spec_rate_order



