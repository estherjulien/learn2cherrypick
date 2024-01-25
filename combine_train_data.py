import pandas as pd
import numpy as np
import pickle
import sys


def combine_data(num_nets, max_l, network_type):
    X = pd.DataFrame(dtype=float)
    Y = pd.Series(dtype=int)
    metadata = pd.DataFrame(dtype=float)
    failed = []
    for i in np.arange(num_nets):
        try:
            with open(f"data/train/inst_results/ML_tree_data_maxL{max_l}_{network_type}_{i}.pickle", "rb") as handle:
                data = pickle.load(handle)
            X = pd.concat([X, data["X"]])
            Y = pd.concat([Y, data["Y"]])
            metadata = pd.concat([metadata, pd.DataFrame(data["metadata"]).transpose()])
        except FileNotFoundError:
            failed.append(i)
            continue
    print(f"(N = {num_nets}, maxL = {max_l}): failed = {failed}")
    with open(f"data/train/ML_tree_data_maxL{max_l}_{network_type}_{num_nets}.pickle", "wb") as handle:
        pickle.dump({"X": X, "Y": Y, "metadata": metadata}, handle)

    print(f"(N = {num_nets}, maxL = {max_l}): finished")


if __name__ == "__main__":
    num_nets = int(sys.argv[1])
    max_l = int(sys.argv[2])
    normal = int(sys.argv[3])
    if normal:
        network_type = "normal"
    else:
        network_type = "LGT"
    combine_data(num_nets, max_l, network_type)
