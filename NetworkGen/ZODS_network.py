import numpy as np
import random
import networkx as nx

'''
Code for generating ZODS network
- Apdated from Remie Janssen's code for the paper (with help of Celine Scornavacca for distances on arcs):
Janssen, R., & Liu, P. (2021). Comparing the topology of phylogenetic network generators. 
Journal of Bioinformatics and Computational Biology, 19(06), 2140012.

- Originally, the method is from the paper:
Zhang, C., Ogilvie, H.A., Drummond, A.J., Stadler, T.: Bayesian inference of species networks from multilocus
sequence data. Molecular biology and evolution 35(2), 504–517 (2018)
'''


def reticulations(G):
    return [v for v in G.nodes() if G.in_degree(v) == 2]


def birth_hyb(time_limit, speciation_rate, hybridization_rate, taxa_goal=None, max_retics=None, distances=True):
    nw = nx.DiGraph()
    nw.add_node(0)
    leaves = {0}
    current_node = 1
    no_of_leaves = 1
    retics = 0

    extra_time = np.random.exponential(1/float(speciation_rate))
    current_time = extra_time
    current_speciation_rate = float(speciation_rate)
    current_hybridization_rate = 0
    rate = current_speciation_rate + current_hybridization_rate

    while (current_time < time_limit and taxa_goal is None) or (taxa_goal is not None and taxa_goal != no_of_leaves):
        if (0 in leaves) or (random.random() < current_speciation_rate / rate):
            # speciate
            splitting_leaf = random.choice(list(leaves))
            if distances:
                nw.add_weighted_edges_from([(splitting_leaf, current_node, 0), (splitting_leaf, current_node + 1, 0)],
                                           weight='length')
            else:
                nw.add_edges_from([(splitting_leaf, current_node), (splitting_leaf, current_node + 1)])
            leaves.remove(splitting_leaf)
            leaves.add(current_node)
            leaves.add(current_node+1)
            current_node += 2
            no_of_leaves += 1
        elif len(leaves) >= 2:
            # Hybridize
            # i.e.: pick two leaf nodes, merge those, and add a new leaf below this hybrid node.
            # can only be two leaves not in a cherry
            # leaves_subset = set()
            # for l in leaves:
            #     for p in nw.predecessors(l):
            #         pl = p
            #     for s in nw.successors(pl):
            #         if s == l:
            #             continue
            #         if s in leaves_subset:
            #             continue
            #         else:
            #             leaves_subset.add(l)

            merging = random.sample(leaves, 2)
            l0 = merging[0]
            l1 = merging[1]
            pl0 = -1
            for p in nw.predecessors(l0):
                pl0 = p
            pl1 = -1
            for p in nw.predecessors(l1):
                pl1 = p
            # If pl0==pl1, the new hybridization results in parallel edges.
            if pl0 != pl1:
                nw.remove_node(l1)
                if distances:
                    nw.add_weighted_edges_from([(pl1, l0, 0), (l0, current_node, 0)], weight='length')
                else:
                    nw.add_edges_from([(pl1, l0), (l0, current_node)])
                leaves.remove(l0)
                leaves.remove(l1)
                leaves.add(current_node)
                current_node += 1
                no_of_leaves -= 1
                retics += 1
        else:
            # not orchard
            print(f"Finished not orchard, current time = {current_time}")
            break

        # extend all pendant edges
        if distances:
            for l in leaves:
                pl = -1
                for p in nw.predecessors(l):
                    pl = p
                nw[pl][l]['length'] += extra_time
        current_speciation_rate = float(speciation_rate*no_of_leaves)
        current_hybridization_rate = float(hybridization_rate * (no_of_leaves * (no_of_leaves - 1))/2)
        rate = current_speciation_rate + current_hybridization_rate
        extra_time = np.random.exponential(1/rate)  # REMIE'S CODE THIS WAS 1/rate
        current_time += extra_time
        if max_retics is not None and retics > max_retics:
            return None, retics, no_of_leaves

    # nothing has happened yet, and there is only one node
    if len(nw) == 1:
        nw.add_edges_from([(0, 1)])
    return nw, retics, no_of_leaves



