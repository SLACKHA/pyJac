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
import itertools

import numpy as np

# Local imports
from .. import utils

#dependencies
have_bitarray = True
try:
    from bitarray import bitarray
except:
    print('bitarray not found, turning off cache-optimization')
    have_bitarray = False

def plot(specs, reacs, consider_thd, fwd_spec_mapping, fwd_rxn_mapping):
    nr = len(reacs)
    nsp = len(specs)
    #plot for visibility
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    arr = np.zeros((nr, nsp + 1, 3))
    arr.fill(1)

    name_map = {sp.name: i for i, sp in enumerate(specs)}
    for rind in range(nr):
        rxn = reacs[rind]
        plot = set(rxn.reac + rxn.prod)
        if consider_thd:
            plot = plot.union(set([x[0] for x in rxn.thd_body_eff] + [rxn.pdep_sp]))
        plot = [name_map[sp] for sp in plot if sp]
        for sp in plot:
            arr[rind, sp] = [0, 0, 0]

    plt.imshow(arr, interpolation='nearest')
    plt.savefig('old.pdf')

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    arr = np.zeros((nr, nsp + 1, 3))
    arr.fill(1)

    name_map = {specs[fwd_spec_mapping[i]].name: i for i, sp in enumerate(fwd_spec_mapping)}
    #print(name_map)
    for i, rind in enumerate(fwd_rxn_mapping):
        rxn = reacs[rind]
        plot = set(rxn.reac + rxn.prod)
        if consider_thd:
            plot = plot.union(set([x[0] for x in rxn.thd_body_eff] + [rxn.pdep_sp]))
        plot = [name_map[sp] for sp in plot if sp]
        for sp in plot:
            arr[i, sp] = [0, 0, 0]

    plt.imshow(arr, interpolation='nearest')
    plt.savefig('new.pdf')

def optimizer_loop_star(args):
    return optimizer_loop(*args)

def optimizer_loop(starting_order, mapping, lookback, improve_cutoff):
    nvar = len(starting_order)
    order = starting_order[:]

    def __get_score(val_mapping, i):
        score = 0
        #start with a blank mapping, and obtain all distinct species
        #that participate in the reactions in the range
        for j in range(max(i - lookback, 0), min(i + lookback + 1, nvar)):
            if i == j:
                continue
            #number that the value in question and the range value share
            count = (val_mapping & mapping[order[j]]).count()
            #number that the value in question does not have, and the
            #range value does, this represents a potential load
            count -= (~val_mapping & mapping[order[j]]).count()

            #scale this by the 
            score += count / float(abs(i - j))

        return score

    def __global_score():
        score = 0
        for i in range(nvar):
            #get the score
            count = __get_score(mapping[order[i]], i)
            score += count

        return score

    #first move any empty entries to the end
    zero_vals = []
    for i in range(nvar):
        if mapping[order[i]].count() == 0:
            zero_vals.append(order[i])
    order = [x for x in order if not x in zero_vals]
    nvar = len(order)
    order += zero_vals

    nvar = next((i for i, val in enumerate(order) if mapping[val].count() == 0), nvar)

    last_improvement = 0
    starting_score = __global_score()
    global_max = starting_score
    global_max_order = order[:]
    while last_improvement < improve_cutoff:
        mincount = None
        mininds = None
        #first scan to see the 'worst' placed reaction
        for i in range(nvar):
            count = __get_score(mapping[order[i]], i)

            if mincount is None or count < mincount:
                mincount = count
                mininds = [i]
            elif mincount is not None and count == mincount:
                mininds += [i]

        moves = []
        for min_ind in mininds: 
            maxcount = None
            maxind = None
            #we now have the 'worst' location selected
            #let's find the 'best place to put it'

            for i in range(nvar):
                if i == min_ind:
                    continue

                 #get the score
                count = __get_score(mapping[order[min_ind]], i)

                if maxcount is None or count > maxcount:
                    maxcount = count
                    maxind = i

            moves.append((min_ind, maxcount, maxind))

        best_move = np.argmax([x[1] for x in moves])
        minind, maxcount, maxind = moves[best_move]
        #now move to the better spot
        order.insert(maxind, order.pop(minind))

        #and compute the score
        score = __global_score()
        if score <= starting_score:
            last_improvement += 1
        else:
            last_improvement = 0
            starting_score = score
        if score > global_max:
            global_max = score
            global_max_order = order[:]
        #print(score, starting_score)
    return global_max, global_max_order

def optimize_cache(specs, reacs, multi_thread,
                    force_optimize, build_path,
                     last_spec, consider_thd=False,
                     improve_cutoff=50,
                     rand_init_tries=10000,
                     lookback=1):
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
    spec_mapping = [bitarray([False for i in range(nr)]) for i in range(nsp)]

    reac_mapping = [bitarray([False for i in range(nsp)]) for i in range(nr)]

    eff_map = [np.zeros(nsp) for i in range(nr)]
    name_map = {sp.name: i for i, sp in enumerate(specs)}
    for rind, rxn in enumerate(reacs):
        for sp in rxn.reac:
            spind = name_map[sp]
            spec_mapping[spind][rind] = True
            reac_mapping[rind][spind] = True
        for sp in rxn.prod:
            spind = name_map[sp]
            spec_mapping[spind][rind] = True
            reac_mapping[rind][spind] = True
        if consider_thd:
            for sp, eff in rxn.thd_body_eff:
                spind = name_map[sp]
                eff_map[rind][spind] = eff

    mapping_list = []
    pool = multiprocessing.Pool(multi_thread if multi_thread else 1)

    fwd_rxn_mapping = [x for x in range(nr)]
    mapping_list.append(fwd_rxn_mapping)
    if rand_init_tries:
        for i in range(rand_init_tries):
            mapping_list.append(np.random.permutation(nr).tolist())
    val = pool.map(optimizer_loop_star, 
            itertools.izip(mapping_list, itertools.repeat(reac_mapping), 
                itertools.repeat(lookback), itertools.repeat(improve_cutoff)))
    fwd_rxn_mapping = val[np.argmax([x[0] for x in val])][1][:]

    fwd_spec_mapping = [i for i in range(nsp) if i != last_spec]
    mapping_list = [fwd_spec_mapping]
    if rand_init_tries:
        for i in range(rand_init_tries):
            mapping_list.append(
                np.random.permutation([i for i in range(nsp) if i != last_spec]).tolist())
    
    val = pool.map(optimizer_loop_star, 
            itertools.izip(mapping_list, itertools.repeat(spec_mapping), 
                itertools.repeat(lookback), itertools.repeat(improve_cutoff)))
    fwd_spec_mapping = val[np.argmax([x[0] for x in val])][1][:] + [last_spec]

    reverse_spec_mapping = [fwd_spec_mapping.index(i) for i in range(len(fwd_spec_mapping))]
    reverse_rxn_mapping = [fwd_rxn_mapping.index(i) for i in range(len(fwd_rxn_mapping))]

    plot(specs, reacs, consider_thd, fwd_spec_mapping, fwd_rxn_mapping)

    specs = specs[fwd_spec_mapping]
    reacs = reacs[fwd_rxn_mapping]
    
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
