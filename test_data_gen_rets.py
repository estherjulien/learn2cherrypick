from NetworkGen.NetworkToTree import *
from NetworkGen.LGT_network import simulation as lgt_simulation
from NetworkGen.normal_network import simulation as normal_simulation
from NetworkGen.ZODS_network import birth_hyb
from NetworkGen.tree_to_newick import *

from joblib import Parallel, delayed
from datetime import datetime
import pandas as pd
import numpy as np
import pickle
import time
import sys

'''
Code used for generating test instances. This file consists of three functions:
- make_test_normal:     generates NORMAL test instances
- make_test_lgt:        generates LGT test instances
- make_test_zods:       generates ZODS test instances

RUN in terminal:
python test_data_gen_rets.py <num. instances> <bool (0/1) for NORMAL instances> <bool (0/1) for LGT instances> 

EXAMPLE:
python test_data_gen_rets.py 16 0 1 0
'''


def make_test_normal(net_num, l, ret, missing_leaves=0, print_failed=False):
    network_gen = "normal"
    tree_info = f"_L{l}_R{ret}_normal"
    if missing_leaves > 0:
        tree_info += f"_PART[{missing_leaves}]"

    # MAKE NETWORK
    st = time.time()
    beta = 1
    distances = True
    n = l - 2 + ret

    # print info
    now = datetime.now().time()
    print(f"JOB {net_num} ({now}): Start creating NETWORK (Normal, L = {l}, R = {ret}, n = {n})")

    while True:
        if l <= 20:
            alpha = np.random.uniform(0.1, 0.5)
        elif l <= 50:
            alpha = np.random.uniform(0.1, 0.3)
        else:
            alpha = np.random.uniform(0.1, 0.2)
        net, ret_num = normal_simulation(n, alpha, 1, beta)
        num_leaves = len(leaves(net))
        if num_leaves == l and ret_num == ret:
            break
        elif print_failed:
            print(f"{network_gen} NETWORK GEN FAILED. r = {ret_num}, l = {num_leaves} ")

    # EXTRACT TREES
    net_nodes = int(len(net.nodes))
    now = datetime.now().time()

    while True:
        print(f"JOB {net_num} ({now}): Start creating TREE SET (Normal, L = {num_leaves}, R = {ret_num})")
        tree_set, tree_lvs, num_unique_leaves = net_to_tree(net, num_trees=None, distances=distances, net_lvs=num_leaves, partial=missing_leaves)
        if num_unique_leaves == num_leaves:
            break

    num_trees = 2 ** ret_num
    tree_to_newick_fun(tree_set, net_num, tree_info=tree_info, network_gen=network_gen)

    # SAVE INSTANCE
    metadata_index = ["network_type", "rets", "nodes", "net_leaves", "chers", "ret_chers", "trees", "n", "alpha",
                      "beta", "missing_lvs_perc", "min_lvs", "mean_lvs", "max_lvs", "runtime"]

    net_cher, net_ret_cher = network_cherries(net)
    min_lvs = min(tree_lvs)
    mean_lvs = np.mean(tree_lvs)
    max_lvs = max(tree_lvs)
    metadata = pd.Series([0, ret_num, net_nodes, num_leaves, len(net_cher)/2, len(net_ret_cher),
                          num_trees, n, alpha, beta,
                          missing_leaves/100, min_lvs, mean_lvs, max_lvs,
                          time.time() - st],
                         index=metadata_index,
                         dtype=float)
    output = {"net": net, "forest": tree_set, "metadata": metadata}
    save_map = "normal"

    with open(
            f"AMBCode/Data/Test/{save_map}/instances/tree_data{tree_info}_{net_num}.pickle", "wb") as handle:
        pickle.dump(output, handle)

    now = datetime.now().time()
    print(f"JOB {net_num} ({now}): FINISHED in {np.round(time.time() - st, 3)}s (Normal, L = {num_leaves}, "
          f"R = {ret_num}, n = {n})")
    return None


def make_test_lgt(net_num, l, ret, num_trees, missing_leaves, print_failed=False):
    network_gen = "LGT"
    tree_info = f"_L{l}_R{ret}_T{num_trees}_LGT"
    if missing_leaves > 0:
        tree_info += f"_PART[{missing_leaves}]"

    # MAKE NETWORK
    st = time.time()
    beta = 1
    distances = True
    n = l - 2 + ret

    # print info
    now = datetime.now().time()
    print(f"JOB {net_num} ({now}): Start creating NETWORK (LGT, L = {l}, R = {ret}, n = {n})")

    while True:
        if l <= 20:
            alpha = np.random.uniform(0.1, 0.5)
        elif l <= 50:
            alpha = np.random.uniform(0.1, 0.3)
        else:
            alpha = np.random.uniform(0.1, 0.2)
        net, ret_num = lgt_simulation(n, alpha, 1, beta)
        num_leaves = len(leaves(net))
        if num_leaves == l and ret_num == ret:
            break
        elif print_failed:
            print(f"{network_gen} NETWORK GEN FAILED. r = {ret_num}, l = {num_leaves} ")

    # EXTRACT TREES
    net_nodes = int(len(net.nodes))
    now = datetime.now().time()

    while True:
        print(
            f"JOB {net_num} ({now}): Start creating TREE SET (LGT, L = {num_leaves}, R = {ret_num}, T = {num_trees})")
        tree_set, tree_lvs, num_unique_leaves = net_to_tree(net, num_trees, distances=distances, net_lvs=num_leaves, partial=missing_leaves)
        if num_unique_leaves == num_leaves:
            break

    tree_to_newick_fun(tree_set, net_num, tree_info=tree_info, network_gen=network_gen)

    # SAVE INSTANCE
    metadata_index = ["network_type", "rets", "nodes", "net_leaves", "chers", "ret_chers", "trees", "n", "alpha",
                      "beta", "missing_lvs_perc", "min_lvs", "mean_lvs", "max_lvs", "runtime"]

    net_cher, net_ret_cher = network_cherries(net)
    min_lvs = min(tree_lvs)
    mean_lvs = np.mean(tree_lvs)
    max_lvs = max(tree_lvs)
    metadata = pd.Series([0, ret_num, net_nodes, num_leaves, len(net_cher)/2, len(net_ret_cher),
                          num_trees, n, alpha, beta,
                          missing_leaves/100, min_lvs, mean_lvs, max_lvs,
                          time.time() - st],
                         index=metadata_index,
                         dtype=float)
    output = {"net": net, "forest": tree_set, "metadata": metadata}
    save_map = "LGT"

    with open(
            f"data/test/{save_map}/instances/tree_data{tree_info}_{net_num}.pickle", "wb") as handle:
        pickle.dump(output, handle)

    now = datetime.now().time()
    print(f"JOB {net_num} ({now}): FINISHED in {np.round(time.time() - st, 3)}s (LGT, L = {num_leaves}, "
          f"R = {ret_num}, T = {num_trees}, n = {n})")
    return None


def make_test_zods(net_num, l, ret, num_trees, missing_leaves, print_failed=False):
    network_gen = "ZODS"
    tree_info = f"_L{l}_R{ret}_T{num_trees}_ZODS"
    if missing_leaves > 0:
        tree_info += f"_PART[{missing_leaves}]"

    st = time.time()
    # MAKE NETWORK
    now = datetime.now().time()
    s_rate = 1.0
    distances = True

    print(f"JOB {net_num} ({now}): Start creating NETWORK (ZODS, L = {l}, R = {ret}, T = {num_trees})")
    while True:
        h_rate = np.random.uniform(0.0001, 0.4)
        net, ret_num, num_leaves = birth_hyb(500, s_rate, h_rate, taxa_goal=l, max_retics=ret)
        if net is not None and len(leaves(net)) == l and ret_num == ret:
            break
        elif print_failed:
            print(f"{network_gen} NETWORK GEN FAILED. r = {ret_num}, l = {num_leaves} ")

    # EXTRACT TREES
    net_nodes = int(len(net.nodes))
    now = datetime.now().time()

    while True:
        print(f"JOB {net_num} ({now}): Start creating TREE SET (ZODS, L = {l}, R = {ret_num}, T = {num_trees})")
        tree_set, tree_lvs, num_unique_leaves = net_to_tree(net, num_trees, distances=distances, net_lvs=num_leaves, partial=missing_leaves)
        if num_unique_leaves == num_leaves:
            break

    tree_to_newick_fun(tree_set, net_num, tree_info=tree_info, network_gen=network_gen)

    # SAVE INSTANCE
    metadata_index = ["network_type", "rets", "nodes", "net_leaves", "chers", "ret_chers", "trees", "h_rate",
                      "missing_lvs_perc", "min_lvs", "mean_lvs", "max_lvs", "runtime"]

    net_cher, net_ret_cher = network_cherries(net)
    min_lvs = min(tree_lvs)
    mean_lvs = np.mean(tree_lvs)
    max_lvs = max(tree_lvs)
    metadata = pd.Series([2, ret_num, net_nodes, num_leaves, len(net_cher)/2, len(net_ret_cher),
                          len(tree_set), h_rate,
                          missing_leaves/100, min_lvs, mean_lvs, max_lvs,
                          time.time() - st],
                         index=metadata_index,
                         dtype=float)
    output = {"net": net, "forest": tree_set, "metadata": metadata}
    save_map = "ZODS"

    with open(
            f"data/test/{save_map}/instances/tree_data{tree_info}_{net_num}.pickle", "wb") as handle:
        pickle.dump(output, handle)

    now = datetime.now().time()
    print(f"JOB {net_num} ({now}): FINISHED in {np.round(time.time() - st, 3)}s (ZODS, L = {num_leaves}, "
          f"R = {ret_num}, T = {num_trees})")
    return None


if __name__ == "__main__":
    num_instances = int(sys.argv[1])
    normal = int(sys.argv[2])
    lgt = int(sys.argv[3])
    missing_leaves = int(sys.argv[4])

    if normal:
        Parallel(n_jobs=-1)(delayed(make_test_normal)(i, l, ret, missing_leaves)
                            for i in np.arange(num_instances)
                            for l in [20, 50, 100]
                            for ret in [5, 6, 7])
    elif lgt:
        Parallel(n_jobs=-1)(delayed(make_test_lgt)(i, l, ret, num_trees, missing_leaves)
                            for i in np.arange(num_instances)
                            for l in [20, 50, 100]
                            for ret in [10, 20, 30]
                            for num_trees in [20, 50, 100])
    else:
        Parallel(n_jobs=-1)(delayed(make_test_zods)(i, l, ret, num_trees, missing_leaves)
                            for i in np.arange(num_instances)
                            for l in [20, 50, 100]
                            for ret in [10, 20, 30]
                            for num_trees in [20, 50, 100])

