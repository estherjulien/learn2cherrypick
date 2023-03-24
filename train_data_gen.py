from NetworkGen.NetworkToTree import *
from NetworkGen.normal_network import simulation as simulation_normal
from NetworkGen.LGT_network import simulation as simulation_LGT

from datetime import datetime
import pandas as pd
import numpy as np
import pickle
import time
import sys

'''
Code used for generating train data.
make_train_data:        - Make a normal network and get data points by calling the function net_to_reduced_trees
net_to_reduced_trees:   - 1. Take all displayed trees from network, and set this as a tree set
                        - 2. While stopping criterion not met: 
                                - reduce a (reticulated) cherry in the tree set and the network
                                - make/update features for each cherry
                                - label all cherries in tree set as: 
                                            0: if cherry (x, y) nor (y, x) is a cherry in the network
                                            1: (x, y) is a cherry in the network
                                            2: (x, y) is a reticulated cherry in the network
                                            3: (y, x) is a reticulated cherry in the network
                        - 3: Store all labelled data
                        
RUN in terminal:
python train_data_gen.py <number of networks> <maxL> <bool (0/1) for data on NORMAL instances>
maxL: maximum number of leaves per network
EXAMPLE:
python train_data_gen.py 10 20
'''


def make_train_data(net_num=0, num_red=100, max_l=100, normal=True, save_network=False):
    # 1. Make a network
    if normal:
        simulation = simulation_normal
        network_type = "normal"
    else:
        simulation = simulation_LGT
        network_type = "LGT"
    # params of LGT generator
    beta = 1
    distances = True

    now = datetime.now().time()
    st = time.time()
    # choose n
    max_ret = 9
    max_n = max_l - 2 + max_ret
    tree_info = f"_maxL{max_l}_{network_type}"
    n = np.random.randint(1, max_n)

    if n < 30:
        alpha = 0.3
    elif n > 50:
        alpha = 0.1
    else:
        alpha = 0.2

    # make network
    network_gen_st = time.time()
    trial = 1
    print(f"JOB {net_num} ({now}): Start creating NETWORK (maxL = {max_l}, n = {n})")
    while True:
        net, ret_num = simulation(n, alpha, 1, beta)
        num_leaves = len(leaves(net))
        if ret_num < max_ret+1 and num_leaves <= max_l:
            break
        if time.time() - network_gen_st > 5*trial:
            n = np.random.randint(1, max_n)
            print(f"JOB {net_num} ({now}): Start creating NEW NETWORK (maxL = {max_l}, n = {n})")
            trial += 1

    if save_network:
        with open(f"test_network_{net_num}.pickle", "wb") as handle:
            pickle.dump(net, handle)

    net_nodes = int(len(net.nodes))
    now = datetime.now().time()
    print(f"JOB {net_num} ({now}): Start creating TREE SET (maxL = {max_l}, L = {num_leaves}, R = {ret_num})")
    num_rets = len(reticulations(net))

    now = datetime.now().time()
    num_trees = 2 ** num_rets

    metadata_index = ["rets", "reductions", "nodes", "net_leaves", "chers", "ret_chers", "trees", "n", "alpha", "beta",
                      "runtime"]

    print(f"JOB {net_num} ({now}): Start creating DATA SET (maxL = {max_l}, L = {num_leaves}, R = {ret_num}, T = {num_trees})")
    X, Y, num_cher, num_ret_cher, tree_set_num = net_to_reduced_trees(net, num_red, num_rets, net_num=net_num, distances=distances,
                                                                                net_lvs=num_leaves)
    print(f"JOB {net_num} ({now}): DATA GENERATION NETWORK FINISHED in {np.round(time.time() - st, 3)}s "
          f"(maxL = {max_l}, L = {num_leaves}, R = {ret_num}, T = {num_trees}, X = {len(X)})")

    metadata = pd.Series([num_rets, num_red, net_nodes, num_leaves, np.mean(num_cher), np.mean(num_ret_cher),
                          np.mean(num_ret_cher), n, alpha, beta, time.time() - st],
                         index=metadata_index,
                         dtype=float)
    output = {"net": net, "X": X, "Y": Y, "metadata": metadata}
    with open(
            f"data/train/inst_results/ML_tree_data{tree_info}_{net_num}.pickle", "wb") as handle:
        pickle.dump(output, handle)


if __name__ == "__main__":
    net_num = int(sys.argv[1])
    max_l = int(sys.argv[2])
    normal = int(sys.argv[3])
    make_train_data(net_num=net_num, max_l=max_l, normal=normal)
