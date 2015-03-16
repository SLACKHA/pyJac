""" Cache Optimizer 

    Reorders loads of rate and species subs to optimize cache hits, etc.
"""

# Python 2 compatibility
from __future__ import division
from __future__ import print_function

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

class score_function(object):
    """
    A class designed to act as a scoring mechanism for the greedy optimizer
    """

    def to_go(self):
        return sum(len(x) for x in self.s_to_r) + sum(len(x) for x in self.r_to_s)

    def load_s_to_r(self, specs, reacs, load_non_participating=False):
        self.r_to_s_save = [set() for i in range(len(reacs))]
        self.s_to_r_save = [set() for i in range(len(specs))]
        for sind, sp in enumerate(specs):
            for rind, rxn in enumerate(reacs):
                if any(sp.name == x for x in rxn.reac + rxn.prod):
                    nu = get_nu(sp, rxn)
                    if nu is not None and (nu != 0 or load_non_participating):
                        self.r_to_s_save[rind].add(sind)
                        self.s_to_r_save[sind].add(rind)

    def __init__(self, specs, reacs, pool_size, load_non_participating=False, pool_size_reset = 0.9):
        # build the map of reactions -> species, species -> reactions
        self.load_s_to_r(specs, reacs, load_non_participating)
        self.restore()
        #get baseline stats
        self.gather_initial_stats(specs, reacs)
        self.pool_size = pool_size
        self.pool_size_reset = pool_size_reset


    def restore(self):
        self.r_to_s = [x.copy() for x in self.r_to_s_save]
        self.s_to_r = [x.copy() for x in self.s_to_r_save]

    def gather_stats(self, the_list):
        raise NotImplementedError

    def gather_initial_stats(self, specs, reacs):
        raise NotImplementedError

    def sp_score(self, current_specs, current_rxns, sp, unused_check = False):
        """
        Score fn for an individual species, overriden in inheriting classes
        """
        raise NotImplementedError

    def rxn_score(self, current_specs, current_rxns, sp, unused_check = False):
        """
        Score fn for an individual rxn, overriden in inheriting classes
        """
        raise NotImplementedError

    def get_sp_score(self, current_specs, current_rxns):
        """
        Evaluates a numerical score for a potential grouping of reactions and species in order to greedily select the next species/reaction

        Returns
        -------
            The index and score of the winning species
        """
        sp_to_add = -1
        spec_add_result = -1
        current_len = len(current_specs) + len(current_rxns)
        temp_set = set(range(len(self.r_to_s))).difference(current_rxns)
        # find the species that shares the most reactions w/ our current list
        for sp in range(len(self.s_to_r)):
            if sp in current_specs:
                continue
            #what adding this species gives directly
            current_score = self.sp_score(current_specs, current_rxns, sp) 
            #what adding this species potentially allows
            possible_score = self.sp_score(current_specs, temp_set, sp)
            #weighted
            result = (current_len / self.pool_size) * current_score + (1.0 - (current_len / self.pool_size)) * possible_score
            if result > spec_add_result and result > 0:
                spec_add_result = result
                sp_to_add = sp
        return sp_to_add, spec_add_result

    def get_rxn_score(self, current_specs, current_rxns):
        """
        Evaluates a numerical score for a potential grouping of reactions and species in order to greedily select the next species/reaction

        Returns
        -------
            The index and score of the winning reaction
        """
        rxn_to_add = -1
        rxn_add_result = -1
        current_len = len(current_specs) + len(current_rxns)
        temp_set = set(range(len(self.s_to_r))).difference(current_specs)
        for rxn in range(len(self.r_to_s)):
            if rxn in current_rxns:
                # skip anything currently in the reaction list
                continue
            # what this reaction gives directly
            current_score = self.rxn_score(current_specs, current_rxns, rxn)
            # what this reaction potentially allows
            possible_score = self.rxn_score(temp_set, current_rxns, rxn)
            #weighted
            result = (current_len / self.pool_size) * current_score + (1.0 - (current_len / self.pool_size)) * possible_score
            if result > rxn_add_result and result > 0:
                rxn_add_result = result
                rxn_to_add = rxn
        return rxn_to_add, rxn_add_result

    def print_stats(self, the_list):
        stats = self.gather_stats(the_list)
        print("Baseline {} loads and {} stores reduced to {} and {}.".format(self.baseline_loads, self.baseline_stores,
                                                                             stats[0], stats[1]))

    def get_list(self, specs, reacs):
        return ([spec for spec in specs], [reac for reac in reacs])

    def update_lists(self, ret_list, spec_list, rxn_list):
        """
        Updates the s->r and r->s mappings based on the current reaction and species lists
        """
        #update the return list
        ret_list.append(self.get_list(spec_list, rxn_list))
        # update the mappings
        for spec in spec_list:
            # get list of reactions that match this species
            my_reacs = [rxn for rxn in rxn_list if rxn in self.s_to_r[spec]]
            if len(my_reacs):
                # remove these reactions from s_to_r
                self.s_to_r[spec] = self.s_to_r[spec].difference(my_reacs)
                for reac in my_reacs:
                    self.r_to_s[reac] = self.r_to_s[reac].difference([spec])

class species_rates_score(score_function):
    def __init__(self, specs, reacs, pool_size, pool_size_reset = 0.9):
        super(species_rates_score, self).__init__(specs, reacs, pool_size)

    def get_list(self, specs, reacs):
        return ([spec for spec in specs], [reac for reac in reacs])

    def update_lists(self, ret_list, spec_list, rxn_list):
        """
        Updates the s->r and r->s mappings based on the current reaction and species lists
        """
        # update the mappings
        for spec in spec_list:
            # get list of reactions that match this species
            my_reacs = [rxn for rxn in rxn_list if rxn in self.s_to_r[spec]]
            if len(my_reacs):
                #update the return list
                ret_list.append(self.get_list([spec], my_reacs))
                # remove these reactions from s_to_r
                self.s_to_r[spec] = self.s_to_r[spec].difference(my_reacs)
                for reac in my_reacs:
                    self.r_to_s[reac] = self.r_to_s[reac].difference([spec])

    def gather_initial_stats(self, specs, reacs):
        self.baseline_loads = len(self.s_to_r) + sum(len(s) for s in self.s_to_r)
        self.baseline_stores = len(self.s_to_r)

    def gather_stats(self, the_list):
        loads = 0
        stores = 0
        for i, c_list in enumerate(the_list):
            for entry in c_list[0]:
                if (i > 0 and entry not in the_list[i - 1][0]) or i == 0:
                    loads += 1
                if ((i < len(the_list) - 1 and entry not in the_list[i + 1][0])
                                        or i == len(the_list) - 1):
                    stores += 1
            for entry in c_list[1]:
                if (i > 0 and entry not in the_list[i - 1][1]) or i == 0:
                    loads += 1
                if ((i < len(the_list) - 1 and entry not in the_list[i + 1][1])
                                        or i == len(the_list) - 1):
                    stores += 1

        return loads, stores

    def rxn_score(self, current_specs, current_rxns, rxn, unused_check = False):
        return sum(1 for s in current_specs if rxn in self.s_to_r[s])

    def sp_score(self, current_specs, current_rxns, sp, unused_check = False):
        return sum(1 for rxn in current_rxns if sp in self.r_to_s[rxn])

class reaction_rates_score(score_function):
    def gather_initial_stats(self, specs, reacs):
        #fwd + rev reacs
        self.baseline_stores = len(reacs) + sum(1 for rxn in reacs if rxn.rev)
        #sum of non-zero nu's for reacs/prods
        self.baseline_loads = sum( sum(1 for spec in rxn.reac) + sum(1 for spec in rxn.prod if rxn.rev) for rxn in reacs)

    def gather_stats(self, the_list):
        loads = 0
        stores = 0
        for i, c_list in enumerate(the_list):
            for entry in c_list[0]:
                if (i > 0 and entry not in the_list[i - 1][0]) or i == 0:
                    loads += 1
            for entry in c_list[1]:
                if (i > 0 and entry not in the_list[i - 1][1]) or i == 0:
                    loads += 1
        stores = len(self.r_to_s)
        return loads, stores

    def get_rxn_score(self, current_specs, current_rxns):
        return -1, -1

    def rxn_score(self, current_specs, current_rxns, rxn):
        return -1

    def sp_score(self, current_specs, current_rxns, sp, unused_check = False):
        if unused_check:
            return -1
        return sum(1 for rxn in current_rxns if self.r_to_s[rxn].issubset(current_specs + [sp])
                and sp in self.r_to_s[rxn])

    def get_all_rxns(self, spec_list):
        # get the list of all reactions possible for this species list
        rxn_list = []
        sp_set = set()
        for sp in spec_list:
            sp_set.add(sp)
            rxn_list.extend([rxn for rxn in range(len(self.r_to_s)) if 
                not rxn in rxn_list and 
                len(self.r_to_s[rxn]) and
                self.r_to_s[rxn].issubset(sp_set)])

        #now order the rxn_list in a good way
        ret_list = [max(rxn_list, key = lambda x : len(self.r_to_s[x]))]
        rxn_list.remove(ret_list[0])
        #now order by most shared with last few
        while len(rxn_list):
            max_rxn = max(rxn_list, key=lambda x:
            sum([len(self.r_to_s[y].intersection(self.r_to_s[x])) / len(self.r_to_s[y]) for y in ret_list[-1:-5]]))
            ret_list.append(max_rxn)
            rxn_list.remove(max_rxn)
        return ret_list

    def update_lists(self, ret_list, spec_list, rxn_list):
        """
        Updates the s->r and r->s mappings based on the current reaction and species lists
        """
        temp_rxn_list = self.get_all_rxns(spec_list)
        super(reaction_rates_score, self).update_lists(ret_list, spec_list, temp_rxn_list)

    def get_list(self, specs, reacs):
        return ([spec for spec in specs], self.get_all_rxns(specs))

class pdep_rates_score(reaction_rates_score):
    def __init__(self, specs, reacs, pool_size, pool_size_reset = 0.9):
        super(pdep_rates_score, self).__init__(specs, reacs, pool_size, True, pool_size_reset)

    def load_s_to_r(self, specs, reacs, load_non_participating = False):
        self.r_to_s_save = [set() for i in range(len(reacs))]
        self.s_to_r_save = [set() for i in range(len(specs))]
        for sind, sp in enumerate(specs):
            for rind, rxn in enumerate(reacs):
                if any(sp.name == x[0] for x in rxn.thd_body):
                    self.r_to_s_save[rind].add(sind)
                    self.s_to_r_save[sind].add(rind)

    def gather_initial_stats(self, specs, reacs):
        self.baseline_loads = sum(len(rxn.thd_body) for rxn in reacs)
        self.baseline_stores = len(reacs)

class jacobian_score(reaction_rates_score):
    def load_s_to_r(self, specs, reacs, load_non_participating=True):
        self.r_to_s_save = [set() for i in range(len(reacs))]
        self.s_to_r_save = [set() for i in range(len(specs))]
        for sind, sp in enumerate(specs):
            for rind, rxn in enumerate(reacs):
                thd_sp = [thd[0] for thd in rxn.thd_body]
                if any(sp.name == x for x in rxn.reac + rxn.prod + thd_sp):
                    nu = get_nu(sp, rxn)
                    if (nu is not None and nu != 0) or (nu is None):
                        self.r_to_s_save[rind].add(sind)
                        self.s_to_r_save[sind].add(rind)

def greedy_optimizer(scorer):
    """
    A method that attempts to optimize cache hit rates via a greedy selection algorithm 
    to choose sets of species / reactions that can be reused in calculation
    """
    pool_size = scorer.pool_size
    pool_size_reset = scorer.pool_size_reset
    return_list = []

    if scorer.to_go() < pool_size:
        pool_size = scorer.to_go()

    rxn_list = []
    max_sp, dummy = max(enumerate(scorer.s_to_r), key=lambda x: len(x[1]))
    sp_list = [max_sp]
    iter_count = 0
    while scorer.to_go() > 0:
        print(iter_count, sum(len(s) for s in scorer.s_to_r) + sum(len(s) for s in scorer.r_to_s))
        iter_count += 1
        #remove any empty ones
        sp_list = [sp for sp in sp_list if len(scorer.s_to_r[sp]) > 0]
        rxn_list = [rxn for rxn in rxn_list if len(scorer.r_to_s[rxn]) > 0]

        sorted_list = None
        #remove worst until less than specified percentage of pool size
        while len(rxn_list) + len(sp_list) > pool_size_reset * pool_size:
            if sorted_list is None:
                sorted_list = scorer.get_list(sp_list, rxn_list)
                sorted_list = (sorted(sorted_list[0], key = lambda x: len(scorer.s_to_r[x])), 
                               sorted(sorted_list[1], key = lambda x: len(scorer.r_to_s[x])))
    
            sp_score = 0
            if len(sorted_list[0]):
                sp_score = len(scorer.s_to_r[sorted_list[0][0]])
            rxn_score = 0
            if len(sorted_list[1]):
                rxn_score = len(scorer.r_to_s[sorted_list[1][0]])

            if sp_score > rxn_score:
                sp_list.remove(sorted_list[0][0])
                sorted_list = (sorted_list[0][1:], sorted_list[1])
            elif rxn_score > 0:
                rxn_list.remove(sorted_list[1][0])
                sorted_list = (sorted_list[0], sorted_list[1][1:])
            else:
                raise Exception

        #now add back new ones
        while len(rxn_list) + len(sp_list) < pool_size:
            sp_ind, sp_score = scorer.get_sp_score(sp_list, rxn_list)
            rxn_ind, rxn_score = scorer.get_rxn_score(sp_list, rxn_list)

            if sp_score > rxn_score:
                sp_list.append(sp_ind)
            elif rxn_score > 0:
                rxn_list.append(rxn_ind)
            elif sp_score > 0:
                sp_list.append(sp_ind)
            else:
                temp = [x for x in range(len(scorer.s_to_r)) if x not in sp_list and len(scorer.s_to_r[x])]
                if not len(temp):
                    break #empty
                max_sp = max(temp, key=lambda x: len(scorer.s_to_r[x]))
                sp_list.append(max_sp)

        #remove any unused ones
        unused_sp = [sp for sp in sp_list if scorer.sp_score(sp_list, rxn_list, sp, True) == 0]
        unused_rxn = [rxn for rxn in rxn_list if scorer.rxn_score(sp_list, rxn_list, rxn, True) == 0]

        if len(unused_sp) + len(unused_rxn) == 0:
            #update and continue
            scorer.update_lists(return_list, sp_list, rxn_list)
        else:
            sp_list = [x for x in sp_list if not x in unused_sp]
            rxn_list = [x for x in rxn_list if not x in unused_rxn]


    scorer.print_stats(return_list)
    return return_list