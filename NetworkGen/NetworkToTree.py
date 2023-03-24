from CPH import *
import networkx as nx
import pandas as pd
import numpy as np
import itertools
import copy

'''
Code for generating the displayed trees from a network
'''


# return reticulation nodes
def reticulations(G):
    return [v for v in G.nodes() if G.in_degree(v) == 2]


# for non-binary give ret number per reticulation node
def reticulations_non_binary(G):
    return [G.in_degree(i)-1 for i in G.nodes if G.in_degree(i) >= 2]


# return leaves from network
def leaves(net):
    return {u for u in net.nodes() if net.out_degree(u) == 0}


# MAKE TREES FROM NETWORK
def net_to_tree(net, num_trees=None, distances=True, net_lvs=0, partial=0):
    # we only consider binary networks here
    tree_set = dict()
    rets = reticulations(net)
    ret_num = len(rets)
    tree_lvs = []
    if ret_num == 0:
        return False, None, None

    if num_trees is None:
        ret_dels_tmp = itertools.product(*[np.arange(2)] * ret_num)
        ret_dels = None
        for opt in ret_dels_tmp:
            opt = np.array(opt).reshape([1, -1])
            try:
                ret_dels = np.vstack([ret_dels, opt])
            except:
                ret_dels = opt
    else:
        ret_dels_set = set()
        its = 0
        while len(ret_dels_set) < num_trees:
            ret_dels_set.add(tuple(np.random.randint(0, 2, ret_num)))
            its += 1
        ret_dels = np.array([list(opt) for opt in ret_dels_set])

    t = 0
    for opt in ret_dels:
        if opt[0] is None:
            continue
        tree = copy.deepcopy(net)
        tree_lvs.append(net_lvs)
        for i in np.arange(ret_num):
            # STEP 1: DELETE ONE OF THE RETICULATION EDGES
            if opt[i] is None:
                continue
            ret = rets[i]
            # check if reticulation still has indegree 2!
            if ret not in tree.nodes:
                # print("Reticulation node not in nodes anymore")
                continue
            if tree.in_degree(ret) < 2:
                continue
            ret_pre_both = list(tree.pred[ret]._atlas.keys())
            ret_pre_del = ret_pre_both[opt[i]]
            # delete reticulation edge
            tree.remove_edge(ret_pre_del, ret)

            # STEP 2: REGRAFTING
            in_out_one_nodes = [n for n in tree.nodes if (tree.in_degree(n) == 1 and tree.out_degree(n) == 1)]
            for n in in_out_one_nodes:
                for p in tree.predecessors(n):
                    pn = p
                for c in tree.successors(n):
                    cn = c
                # predecessor length
                try:
                    pre_len = tree.edges[(pn, n)]["length"]
                except KeyError:
                    pre_len = tree.edges[(pn, n)]["lenght"]
                # successor length
                try:
                    succ_len = tree.edges[(n, cn)]["length"]
                except KeyError:
                    succ_len = tree.edges[(n, cn)]["lenght"]
                # remove node
                tree.remove_node(n)
                # add edge
                tree.add_edge(pn, cn, length=pre_len + succ_len)
            # STEP 3: DELETE "NEW LEAVES"
            while True:
                wrong_new_leaves = leaves(tree).difference(leaves(net))
                if not wrong_new_leaves:
                    break
                for n in wrong_new_leaves:
                    tree.remove_node(n)

            # STEP 4: REROOTING
            if tree.out_degree(0) == 1:
                # print("Rerooting")
                for c in tree.successors(0):
                    child = c
                tree.remove_node(0)
                tree = nx.relabel_nodes(tree, {child: 0})

        # if partial, delete random leaves from tree
        if max([tree.in_degree(l) for l in tree.nodes]) > 1:
            print("Bad tree before partial")
            continue

        if partial > 0:
            ms_perc = float(partial)/100
            lvs = leaves(tree)
            # delete nodes with outdegree zero and not part of leave
            for l in lvs:
                if tree_lvs[-1] <= 2 or np.random.rand() > ms_perc:
                    continue
                # regraft parent
                tree_lvs[-1] -= 1
                for p in tree.predecessors(l):
                    pl = p
                if pl == 0:
                    # PARENT IS ROOT, REROOTING
                    tree.remove_node(l)
                    for c in tree.successors(0):
                        child = c
                    tree.remove_node(0)
                    tree = nx.relabel_nodes(tree, {child: 0})
                    continue

                for g in tree.predecessors(pl):
                    gl = g
                for s in tree.successors(pl):
                    if s == l:
                        continue
                    sl = s

                # add new edge
                try:
                    len_1 = tree.edges[(gl, pl)]["length"]
                except KeyError:
                    len_1 = tree.edges[(gl, pl)]["lenght"]

                try:
                    len_2 = tree.edges[(pl, sl)]["length"]
                except KeyError:
                    len_2 = tree.edges[(pl, sl)]["lenght"]

                tree.add_edge(gl, sl, length=len_1 + len_2)
                # remove parent and leaf node
                tree.remove_node(l)
                tree.remove_node(pl)

        add_node_attributes(tree, distances=distances, root=0)
        if max([tree.in_degree(l) for l in tree.nodes]) == 1:
            tree_set[t] = tree
            t += 1
        else:
            print("Bad tree after partial")

    if partial > 0:
        unique_leaves = leaves(tree_set[0])
        for t, tree in tree_set.items():
            if t == 0:
                continue
            unique_leaves = leaves(tree).union(unique_leaves)
        num_unique_leaves = len(unique_leaves)
    else:
        num_unique_leaves = len(leaves(tree_set[0]))

    return tree_set, tree_lvs, num_unique_leaves


def net_to_reduced_trees(net, num_red=1, num_rets=0, net_num=0, distances=True, net_lvs=None):
    # extract trees from network
    if num_rets == 0:
        tree = deepcopy(net)
        add_node_attributes(tree, distances=distances, root=0)
        tree_set = {0: tree}
    else:
        tree_set, _, _ = net_to_tree(net, distances=distances, net_lvs=net_lvs)

    # make network and forest environments
    net_env = deepcopy(PhN(net))
    init_forest_env = Input_Set(tree_set=tree_set, leaves=net_env.leaves)
    forest_env = deepcopy(init_forest_env)
    # get cherries from network and forest
    net_cher, net_ret_cher = network_cherries(net_env.nw)
    reducible_pairs = forest_env.find_all_pairs()

    # output information
    num_cher = [len(net_cher)]
    num_ret_cher = [len(net_ret_cher)]
    tree_set_num = [len(forest_env.trees)]

    # features
    features = Features(reducible_pairs, forest_env.trees, root=0)

    # create input X and output Y data
    X = deepcopy(features.data)
    # change index of X
    X_index = [f"{c}_{net_num}" for c in X.index]
    X.index = X_index
    CPS = []
    Y = cherry_labels(net_cher, net_ret_cher, list(reducible_pairs), X.index, net_num)
    # now, reduce tree_set and net at the same time to get labelled data!
    for r in np.arange(num_red):
        triv_picked = False
        # pick random cherry
        cherry_in_all = {c for c, trees in reducible_pairs.items() if len(trees) == len(tree_set)}.intersection(net_cher)
        pickable_chers = net_ret_cher.intersection(set(reducible_pairs))
        if cherry_in_all:    # if any cherry in all, reduce that one first
            chosen_cherry = list(cherry_in_all)[np.random.choice(len(cherry_in_all))]
        elif net_cher:  # otherwise, pick trivial cherry, with relabelling
            chosen_cherry = list(net_cher)[np.random.choice(len(net_cher))]
            # check if we need to relabel
            try:
                triv_check = any([chosen_cherry[0] in tree.leaves for t, tree in forest_env.trees.items() if t not in reducible_pairs[chosen_cherry]])
            except:
                print("CHERRY IN NETWORK NOT A CHERRY IN THE FOREST")
                break
            if triv_check:
                triv_picked = True
            else:
                triv_picked = False
        else:  # reticulate cherries
            try:
                chosen_cherry = list(pickable_chers)[np.random.choice(len(pickable_chers))]
            except ValueError:
                print("NO PICKABLE CHERRIES")
                break
        CPS.append(chosen_cherry)

        if triv_picked:
            reducible_pairs, merged_cherries = forest_env.relabel_trivial(*chosen_cherry, reducible_pairs)
            features.relabel_trivial_features(*chosen_cherry, reducible_pairs, merged_cherries, forest_env.trees)
        # update some features before picking
        features.update_cherry_features_before(chosen_cherry, reducible_pairs, forest_env.trees)
        parent_cherry = forest_env.get_parent_cherry(*chosen_cherry, reducible_pairs[chosen_cherry])
        # reduce trees with chosen cherry
        new_reduced = forest_env.reduce_pair_in_all(chosen_cherry, reducible_pairs=reducible_pairs)
        forest_env.update_node_comb_length(*chosen_cherry, new_reduced)

        if any([any([trees.nw.in_degree(n) == 2 for n in trees.nw.nodes]) for t, trees in
                forest_env.trees.items()]):
            print("RET HAPPENED")
            break

        reducible_pairs = forest_env.update_reducible_pairs(reducible_pairs, new_reduced)
        if len(forest_env.trees) == 0:
            break
        # update features after picking
        features.update_cherry_features_after(chosen_cherry, reducible_pairs, forest_env.trees, new_reduced, parent_cherry)
        net_env.reduce_pair(*chosen_cherry)
        net_cher, net_ret_cher = network_cherries(net_env.nw)

        # output information
        num_cher += [len(net_cher)/2]
        num_ret_cher += [len(net_ret_cher)]
        tree_set_num += [len(forest_env.trees)]

        # in and output cherries
        X_new = deepcopy(features.data)
        # change index of X
        X_index = [f"{c}_{net_num}" for c in X_new.index]
        X_new.index = X_index
        Y_new = cherry_labels(net_cher, net_ret_cher, list(reducible_pairs), X_new.index, net_num)

        X = pd.concat([X, X_new])
        Y = pd.concat([Y, Y_new])

    return X, Y, num_cher, num_ret_cher, tree_set_num


# FIND CHERRIES AND RETICULATED CHERRIES
def network_cherries(net):
    cherries = set()
    retic_cherries = set()
    lvs = leaves(net)

    for l in lvs:
        for p in net.pred[l]:
            if net.out_degree(p) > 1:
                for cp in net.succ[p]:
                    if cp == l:
                        continue
                    if cp in lvs:
                        cherries.add((l, cp))
                        cherries.add((cp, l))
                    elif net.in_degree(cp) > 1:
                        for ccp in net.succ[cp]:
                            if ccp in lvs:
                                retic_cherries.add((ccp, l))

    return cherries, retic_cherries


def tree_cherries(tree_set):
    cherries = set()
    reducible_pairs = dict()
    t = 0
    for tree in tree_set.values():
        lvs = leaves(tree)

        for l in lvs:
            for p in tree.pred[l]:
                if tree.out_degree(p) > 1:
                    for cp in tree.succ[p]:
                        if cp == l:
                            continue
                        if cp in lvs:
                            cherry = (l, cp)
                            cherries.add(cherry)
                            cherries.add(cherry[::-1])

                            # add tree to cherry
                            if cherry not in reducible_pairs:
                                reducible_pairs[cherry] = {t}
                                reducible_pairs[cherry[::-1]] = {t}
                            else:
                                reducible_pairs[cherry].add(t)
                                reducible_pairs[cherry[::-1]].add(t)
        t += 1
    return cherries, reducible_pairs


# check if cherry is reducible
def is_cherry(tree, x, y):
    lvs = leaves(tree)
    if (x not in lvs) or (y not in lvs):
        return False
    # tree, so no reticulations
    px = tree.pred[x]._atlas.keys()
    py = tree.pred[y]._atlas.keys()
    return px == py


def is_ret_cherry(net, x, y):
    for p in net.pred[y]:
        if net.out_degree(p) > 1:
            for cp in net.succ[p]:
                if cp == y:
                    continue
                if net.in_degree(cp) > 1:
                    for ccp in net.succ[cp]:
                        if ccp == x:
                            return True
    return False


# CHERRY LABELS
def cherry_labels(net_cher, net_ret_cher, tree_cher, index, num_net=0):
    # LABELS
    df_labels = pd.DataFrame(0, index=index, columns=np.arange(4), dtype=np.int8)
    for c in tree_cher:
        # cherry in network
        if c in net_cher:
            df_labels.loc[f"{c}_{num_net}", 1] = 1
        elif c in net_ret_cher:
            df_labels.loc[f"{c}_{num_net}", 2] = 1
        elif c[::-1] in net_ret_cher:
            df_labels.loc[f"{c}_{num_net}", 3] = 1
        else:
            df_labels.loc[f"{c}_{num_net}", 0] = 1
    return df_labels
