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
        self.restore()

    def __init__(self, specs, reacs, pool_size, load_non_participating=False):
        # build the map of reactions -> species, species -> reactions
        self.load_s_to_r(specs, reacs, load_non_participating)
        #get baseline stats
        self.gather_initial_stats(specs, reacs)
        self.pool_size = pool_size

    def restore(self):
        self.r_to_s = [x.copy() for x in self.r_to_s_save]
        self.s_to_r = [x.copy() for x in self.s_to_r_save]

    def get_rxn_score(self, current_specs, current_rxns):
        """
        Evaluates a numerical score for a potential grouping of reactions and species in order to greedily select the next species/reaction

        Returns
        -------
            The index and score of the winning reaction
        """
        raise NotImplementedError

    def get_sp_score(self, current_specs, current_rxns):
        """
        Evaluates a numerical score for a potential grouping of reactions and species in order to greedily select the next species/reaction

        Returns
        -------
            The index and score of the winning species
        """
        raise NotImplementedError

    def update_lists(self, spec_list, rxn_list):
        """
        Updates the s->r and r->s mappings based on the current reaction and species lists
        """
        # update the lists
        for spec in spec_list:
            # get list of reactions that match this species
            my_reacs = [rxn for rxn in rxn_list if rxn in self.s_to_r[spec]]
            if len(my_reacs):
                # remove these reactions from s_to_r
                self.s_to_r[spec] = self.s_to_r[spec].difference(my_reacs)
                for reac in my_reacs:
                    self.r_to_s[reac] = self.r_to_s[reac].difference([spec])

    def gather_stats(self, the_list):
        raise NotImplementedError

    def gather_initial_stats(self, specs, reacs):
        raise NotImplementedError

    def print_stats(self, the_list):
        stats = self.gather_stats(the_list)
        print("Baseline {} loads and {} stores reduced to {} and {}.".format(self.baseline_loads, self.baseline_stores,
                                                                             stats[0], stats[1]))


class species_rates_score(score_function):
    def __init__(self, specs, reacs, pool_size):
        super(species_rates_score, self).__init__(specs, reacs, pool_size, True)

    def gather_initial_stats(self, specs, reacs):
        self.baseline_loads = len(self.s_to_r) + sum(len(s) for s in self.s_to_r)
        self.baseline_stores = len(self.s_to_r)

    def gather_stats(self, the_list):
        loads = 0
        stores = 0
        for i, c_list in enumerate(the_list):
            for entry in c_list:
                if (i > 0 and entry not in the_list[i - 1]) or i == 0:
                    loads += 1
                if entry[0] == 's' and ((i < len(the_list) - 1 and entry not in the_list[i + 1])
                                        or i == len(the_list) - 1):
                    stores += 1

        return loads, stores

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
        temp_set = set(range(len(self.s_to_r)))
        for rxn in range(len(self.r_to_s)):
            if rxn in current_rxns:
                # skip anything currently in the reaction list
                continue
            # what this reaction gives directly
            current_score = sum(1 for s in current_specs if rxn in self.s_to_r[s])
            # what this reaction potentially allows
            possible_score = sum(1 for s in temp_set if rxn in self.s_to_r[s])
            #weighted
            result = (current_len / self.pool_size) * current_score + (1.0 - (current_len / self.pool_size)) * possible_score
            if result > rxn_add_result:
                rxn_add_result = result
                rxn_to_add = rxn
        return rxn_to_add, rxn_add_result

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
        temp_set = set(range(len(self.r_to_s)))
        # find the species that shares the most reactions w/ our current list
        for sp in range(len(self.s_to_r)):
            if sp in current_specs:
                continue
            #what adding this species gives directly
            current_score = sum(1 for rxn in current_rxns if any(spec in self.r_to_s[rxn] for spec in current_specs + [sp]))
            #what adding this species potentially allows
            possible_score = sum(1 for rxn in temp_set if any(spec in self.r_to_s[rxn] for spec in current_specs + [sp]))
            #weighted
            result = (current_len / self.pool_size) * current_score + (1.0 - (current_len / self.pool_size)) * possible_score
            if result > spec_add_result:
                spec_add_result = result
                sp_to_add = sp
        return sp_to_add, spec_add_result

class reaction_rates_score(score_function):
    def gather_initial_stats(self, specs, reacs):
        #fwd + rev reacs
        self.baseline_stores = len(reacs) + sum(1 for rxn in reacs if rxn.rev)
        #sum of non-zero nu's for reacs/prods
        self.baseline_loads = sum( sum(1 for spec in rxn.reacs) + sum(1 for spec in rxn.prods if rxn.rev) for rxn in reacs)

    def gather_stats(self, the_list):
        loads = 0
        stores = 0
        for i, c_list in enumerate(the_list):
            for entry in c_list:
                if (i > 0 and entry not in the_list[i - 1]) or i == 0:
                    loads += 1
        stores = len(self.r_to_s)
        return loads, stores

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
        temp_set = set(range(len(self.r_to_s)))
        # find the species that shares the most reactions w/ our current list
        for sp in range(len(self.s_to_r)):
            if sp in current_specs:
                continue
            #what adding this species gives directly
            current_score = sum(1 for rxn in current_rxns if self.s_to_r[rxn].issubset(current_specs + sp))
            #what adding this species potentially allows
            possible_score = sum(1 for rxn in temp_set if self.s_to_r[rxn].issubset(current_specs + sp))
            #weighted
            result = (current_len / self.pool_size) * current_score + (1.0 - (current_len / self.pool_size)) * possible_score
            if result > spec_add_result:
                spec_add_result = result
                sp_to_add = sp
        return sp_to_add, spec_add_result

    def get_all_rxns(self, spec_list):
        # get the list of all reactions possible for this species list
        rxn_list = []
        for rxn in range(len(self.r_to_s)):
            if set(self.r_to_s[rxn]).issubset(spec_list):
                rxn_list.append(rxn)

        return rxn_list

    def update_lists(self, spec_list, rxn_list):
        """
        Updates the s->r and r->s mappings based on the current reaction and species lists
        """
        temp_rxn_list = self.get_all_rxns(spec_list)
        super(reaction_rates_score, self).update_lists(spec_list, temp_rxn_list)


class pdep_rates_score(reaction_rates_score):
    def __init__(self, specs, reacs):
        reacs = [r for r in reacs if r.pdep or r.thd]
        super(pdep_rates_score, self).__init__(specs, reacs, pool_size)

    def load_s_to_r(self, specs, reacs, load_non_participating = False):
        self.r_to_s_save = [set() for i in range(len(reacs))]
        self.s_to_r_save = [() for i in range(len(specs))]
        for sind, sp in enumerate(specs):
            for rind, rxn in enumerate(reacs):
                if any(sp.name == x[0] for x in rxn.thd_body):
                    self.r_to_s_save[rind].append(sind)
                    self.s_to_r_save[sind].append(rind)

    def gather_initial_stats(self, specs, reacs):
        self.baseline_loads = sum(len(rxn.thd_body) for rxn in reacs)
        self.baseline_stores = len(reacs)

def __get_list(reacs, specs):
    return [('r', reac) for reac in reacs] + [('s', spec) for spec in specs]

def __update_list(the_list, reacs, specs):
    the_list.append(__get_list(reacs, specs))

def greedy_optimizer(scorer, pool_size_reset = 0.9):
    pool_size = scorer.pool_size
    return_list = []

    rxn_list = []
    max_sp, dummy = max(enumerate(scorer.s_to_r), key=lambda x: len(x[1]))
    sp_list = [max_sp]
    #first pass, need to fill up entire pool
    while len(rxn_list) + len(sp_list) < pool_size:
        sp_ind, sp_score = scorer.get_sp_score(sp_list, rxn_list)
        rxn_ind, rxn_score = scorer.get_rxn_score(sp_list, rxn_list)

        if sp_score > rxn_score:
            sp_list.append(sp_ind)
        elif rxn_score > 0:
            rxn_list.append(rxn_ind)
        else:
            sp_list.append(sp_ind)

    __update_list(return_list, rxn_list, sp_list)
    scorer.update_lists(sp_list, rxn_list)

    while sum(len(s) for s in scorer.s_to_r) > 0:
        #remove any empty ones
        sp_list = [sp for sp in sp_list if len(scorer.s_to_r[sp]) > 0]
        rxn_list = [rxn for rxn in rxn_list if len(scorer.r_to_s[rxn]) > 0]

        sorted_list = None
        #remove worst until less than specified percentage of pool size
        while len(rxn_list) + len(sp_list) > pool_size_reset * pool_size:
            if sorted_list is None:
                sorted_list = __get_list(rxn_list, sp_list)
                sorted_list = sorted(sorted_list, key = lambda x: len(scorer.s_to_r[x[1]]) if x[0] == 's' else len(scorer.r_to_s[x[1]]))

            first = sorted_list[0]
            if first[0] == 's':
                sp_list.remove(first[1])
            else:
                rxn_list.remove(first[1])

            sorted_list = sorted_list[1:]

        #now add back new ones
        while len(rxn_list) + len(sp_list) < pool_size:
            sp_ind, sp_score = scorer.get_sp_score(sp_list, rxn_list)
            rxn_ind, rxn_score = scorer.get_rxn_score(sp_list, rxn_list)

            if sp_score > rxn_score:
                sp_list.append(sp_ind)
            elif rxn_score > 0:
                rxn_list.append(rxn_ind)
            else:
                sp_list.append(sp_ind)

            #update and continue
            __update_list(return_list, rxn_list, sp_list)
            scorer.update_lists(sp_list, rxn_list)

    scorer.print_stats(return_list)