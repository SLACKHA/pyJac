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
from operator import itemgetter

# Local imports
from CUDAParams import Jacob_Unroll, ResetOnJacUnroll
from CParams import C_Jacob_Unroll
import utils
import numpy as np

LOOKBACK=1
USE_THD=True

#dependencies
have_bitarray = True
try:
    from bitarray import bitarray
except:
    print('bitarray not found, turning off cache-optimization')
    have_bitarray = False

def optimize_cache(specs, reacs, multi_thread,
                    force_optimize, build_path,
                     last_spec):
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
            if reverse_spec_mapping[last_spec] != len(specs) - 1:
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

    nsp = len(specs)
    nr = len(reacs)

    last_name = specs[last_spec].name

    #now generate our mappings
    sdummy = [False for i in range(nr)]
    spec_mapping = [bitarray(sdummy) for i in range(nsp)]

    rdummy = [False for i in range(nsp)]
    reac_mapping = [bitarray(rdummy) for i in range(nr)]

    rxn_to_sp = [[] for i in range(nr)]
    sp_to_rxn = [[] for i in range(nsp)]

    def no_dupe_add(thelist, item):
        if not item in thelist:
            thelist.append(item)

    eff_map = [np.zeros(nsp) for i in range(nr)]
    name_map = {sp.name: i for i, sp in enumerate(specs)}
    for rind, rxn in enumerate(reacs):
        for sp in rxn.reac:
            spind = name_map[sp]
            spec_mapping[spind][rind] = True
            reac_mapping[rind][spind] = True
            no_dupe_add(rxn_to_sp[rind], spind)
            no_dupe_add(sp_to_rxn[spind], rind)
        for sp in rxn.reac:
            spind = name_map[sp]
            spec_mapping[spind][rind] = True
            reac_mapping[rind][spind] = True
            no_dupe_add(rxn_to_sp[rind], spind)
            no_dupe_add(sp_to_rxn[spind], rind)
        if USE_THD:
            for sp, eff in rxn.thd_body_eff:
                spind = name_map[sp]
                eff_map[rind][spind] = eff
    
    def get_rxn_tiebreak(rind, rxn_to_sp, spec_mapping):
        bt = bitarray(sdummy)
        for sp in rxn_to_sp[rind]:
            bt |= spec_mapping[sp]
        return bt.count()

    def update_spec_mapping(i, mapvals, listvals):
        for i2 in listvals[i]:
            mapvals[i2][i] = False

    def update_reac_mapping(i, mapvals, listvals):
        for i2 in listvals[i]:
            mapvals[i][i2] = False

    maxcount = None
    ind = None
    # select the first reaction as the one with the species 
    #that participate in the most distinct reactions
    ind = max(range(nr), key=lambda x: get_rxn_tiebreak(x, rxn_to_sp, spec_mapping))

    #get the updating spec mapping
    #used to determine all distinct reactions for species in a reaction
    #counting only those reactions that haven't been selected yet
    updating = [x.copy() for x in spec_mapping]
    update_spec_mapping(ind, updating, rxn_to_sp)
    reacs_left = [i for i in range(nr) if i != ind]
    fwd_rxn_mapping = [ind]
    while len(reacs_left):
        #the next reaction is the one that best matches the previous one
        #in case of a tie a tie breaker is used:
        #     the number of of the distinct reactions the species in the reaction are in 
        maxcount = None
        ind = None
        last_tiebreak = None
        for insert_index in range(1, len(fwd_rxn_mapping) + 1):
            start = max(insert_index - LOOKBACK, 0)
            for rind in reacs_left:
                mapping = bitarray(rdummy)
                for last_ind in range(start, insert_index):
                    mapping = mapping | reac_mapping[fwd_rxn_mapping[last_ind]]
                #number of species shared in by the reactions
                #minus the number that are in this candidate and not the last reaction
                count = ((reac_mapping[rind] | mapping).count() - 
                            ((reac_mapping[rind] ^ mapping) & reac_mapping[rind]).count())

                if USE_THD:
                    non_zero = np.where(eff_map[rind] != 0.)
                    #for third body efficiency reactions
                    #if the previous reaction is also has third body efficiencies
                    #add the number of efficiencies shared exactly
                    #minus the number unshared
                    for last_ind in range(start, insert_index):
                        diff = eff_map[rind] - eff_map[fwd_rxn_mapping[last_ind]]
                        non_shared = np.count_nonzero(diff)
                        shared = np.where(diff[non_zero] == 0.)[0].shape[0]
                        count += shared - non_shared

                divisor = reac_mapping[rind].count()
                if USE_THD and reacs[rind].thd_body_eff:
                    divisor += non_zero[0].shape[0]
                count = 0 if divisor == 0 else float(count) / float(divisor)

                if maxcount is None or count >= maxcount:
                    #compute tiebreak score
                    tiebreak = get_rxn_tiebreak(rind, rxn_to_sp, updating)
                    winner = True
                    if last_tiebreak is not None and count == maxcount and tiebreak <= last_tiebreak:
                        winner = False
                    if winner:
                        insert = insert_index
                        ind = rind
                        maxcount = count
                        last_tiebreak = tiebreak

        if maxcount == 0:
            insert = -1
        #print (insert, maxcount, ind, len(fwd_rxn_mapping))
        #add the winner to the list
        fwd_rxn_mapping.insert(insert, ind)
        update_spec_mapping(ind, updating, rxn_to_sp)
        reacs_left.remove(ind)

    #ok, we now have a reordered reaction list
    #let's take a whack at the species

    # select the first species as the one in the first reaction
    #that participate in the most reactions
    ind = max(rxn_to_sp[fwd_rxn_mapping[0]], key=lambda x: reac_mapping[x].count() *
                                                    (1 if x != last_spec else 0))

    updating = [x.copy() for x in spec_mapping]
    fwd_spec_mapping = [ind]
    specs_left = [i for i in range(nsp) if i != ind]
    update_reac_mapping(ind, updating, sp_to_rxn)
    #print()
    while len(specs_left) > 1:
        maxcount = None
        ind = None
        last_tiebreak = None
        for insert_index in range(1, len(fwd_spec_mapping) + 1):
            start = max(insert_index - LOOKBACK, 0)
            for spind in specs_left:
                if spind == last_spec:
                    continue
                mapping = bitarray(sdummy)
                for last_ind in range(start, insert_index):
                    mapping = mapping | spec_mapping[fwd_spec_mapping[last_ind]]
                #number of reactions shared with the last species
                #minus the number that are not
                count = ((spec_mapping[spind] | mapping).count() - 
                            ((spec_mapping[spind] ^ mapping) & spec_mapping[spind]).count())

                divisor = spec_mapping[spind].count()
                count = 0 if divisor == 0 else float(count) / float(divisor)

                if maxcount is None or count >= maxcount:
                    #compute tiebreak score
                    tiebreak = updating[spind].count()
                    winner = True
                    if last_tiebreak is not None and count == maxcount and tiebreak <= last_tiebreak:
                        winner = False
                    if winner:
                        insert = insert_index
                        ind = spind
                        maxcount = count
                        last_tiebreak = tiebreak

        if maxcount == 0:
            insert = -1
        #print (specs[ind].name, insert, maxcount, last_tiebreak)
        #add the winner to the list
        fwd_spec_mapping.insert(insert, ind)
        update_reac_mapping(ind, updating, sp_to_rxn)
        specs_left.remove(ind)

    fwd_spec_mapping.append(last_spec)

    reverse_spec_mapping = [fwd_spec_mapping.index(i) for i in range(len(fwd_spec_mapping))]
    reverse_rxn_mapping = [fwd_rxn_mapping.index(i) for i in range(len(fwd_rxn_mapping))]

    #print()
    plot = True
    if plot:
        #plot for visibility
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        arr = np.zeros((nr, nsp + 1, 3))
        arr.fill(1)

        for rind in range(nr):
            rxn = reacs[rind]
            plot = set(rxn.reac + rxn.prod)
            #if USE_THD:
            #    plot = plot.union(set([x[0] for x in rxn.thd_body_eff] + [rxn.pdep_sp]))
            plot = [name_map[sp] for sp in plot if sp]
            for sp in plot:
                arr[rind, sp] = [0, 0, 0]

        loads = 0
        for rind in range(nr - 1):
            for sp in range(nsp):
                if arr[rind, sp, 0] > arr[rind + 1, sp, 0]:
                    loads += 1

        local = 0
        for rind in range(nr):
            for sp in range(nsp - 1):
                if arr[rind, sp, 0] < arr[rind, sp + 1, 0]:
                    local += 1

        #print (loads, local)

        plt.imshow(arr, interpolation='nearest')
        plt.savefig('old.pdf')

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        arr = np.zeros((nr, nsp + 1, 3))
        arr.fill(1)

        name_map = {specs[sp].name: i for i, sp in enumerate(fwd_spec_mapping)}
        for rind in fwd_rxn_mapping:
            rxn = reacs[rind]
            plot = set(rxn.reac + rxn.prod)
            #if USE_THD:
            #    plot = plot.union(set([x[0] for x in rxn.thd_body_eff] + [rxn.pdep_sp]))
            plot = [name_map[sp] for sp in plot if sp]
            for sp in plot:
                arr[rind, sp] = [0, 0, 0]

        loads = 0
        for rind in range(nr - 1):
            for sp in range(nsp):
                if arr[rind, sp, 0] > arr[rind + 1, sp, 0]:
                    loads += 1

        local = 0
        for rind in range(nr):
            for sp in range(nsp - 1):
                if arr[rind, sp, 0] < arr[rind, sp + 1, 0]:
                    local += 1

        #print (loads, local)

        plt.imshow(arr, interpolation='nearest')
        plt.savefig('new.pdf')


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
