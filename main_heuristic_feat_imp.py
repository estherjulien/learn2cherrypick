from datetime import datetime
import pandas as pd
import numpy as np
import copy
import sys
import csv
import warnings

from main_heuristic import run_heuristic
warnings.simplefilter(action='ignore', category=FutureWarning)

'''
Code consisting of main run file and two functions:
- run_heuristic:            1. open tree set and make CPH.PhT environment for each tree
                            2. run cherry picking heuristic (CPH)
                            3. return results
- run_main:                 run CPH with four "PickNextCherry" methods:
                                1. ML
                                2. TrivialML
                                3. Rand
                                4. TrivialRand

RUN in terminal:
python main_heuristic.py <instance num.> <ML model name> <leaf number> <bool (0/1) for exact input> <option>
option: 
if exact input = 0:
    option = reticulation number
else:
    option = forest size
EXAMPLE: 
python main_heuristic.py 0 N10_maxL100_random_balanced 20 0 50
'''


def run_main(i, L, R, forest_size=None, network_type=None,
             repeats=1, time_limit=None, full_leaf_set=True, progress=False, file_name=None):
    if network_type == "normal":
        file_info = f"L{L}_R{R}_normal"
    elif network_type == "ZODS":
        file_info = f"L{L}_R{R}_T{forest_size}_ZODS"
    else:
        file_info = f"L{L}_R{R}_T{forest_size}_LGT"
        network_type = "LGT"

    # save results
    columns = ["FeatImp", "UB"]
    score = pd.DataFrame(
        index=pd.MultiIndex.from_product([[i], ["RetNum", "Time"], np.arange(repeats)]),
        columns=columns, dtype=float)

    # INSTANCE
    tree_set_newick = f"data/test/{network_type}/newick/tree_set_newick_{file_info}_{i}.txt"

    # FEATURE IMPORTANCE HEURISTIC
    ret_score, time_score, seq_fi = run_heuristic(
        tree_set_newick=tree_set_newick,
        inst_num=i,
        repeats=1,
        time_limit=time_limit,
        pick_feat_imp=True,
        relabel=True,
        problem_type="FeatImp",
        full_leaf_set=full_leaf_set,
        progress=progress)

    for r, ret in ret_score.items():
        score.loc[i, "RetNum", r]["FeatImp"] = copy.copy(ret)
        score.loc[i, "Time", r]["FeatImp"] = copy.copy(time_score[r])
    fi_ret = int(min(score.loc[i, "RetNum"]["FeatImp"]))
    fi_time = score.loc[i, "Time", 0]["FeatImp"]

    # upper bound of ret
    idx = pd.IndexSlice
    env_info_file = f"data/test/{network_type}/instances/tree_data_{file_info}_{i}.pickle"
    env_info = pd.read_pickle(env_info_file)
    ub_ret = int(env_info["metadata"]["rets"])
    score.loc[idx[i, "RetNum", :], "UB"] = ub_ret

    # print results
    if progress:
        print()
        print("FINAL RESULTS\n"
              f"Instance = {i} \n"
              "RETICULATIONS\n"
              f"FeatImp       = {fi_ret}\n"
              f"Reference     = {ub_ret}\n")
    else:
        print(i, fi_ret, ub_ret, fi_time)

    # SAVE DATAFRAMES
    # scores
    score.dropna(axis=0, how="all").to_pickle(f"data/results/inst_results/heuristic_scores_{file_name}_{i}.pickle")


if __name__ == "__main__":
    # TEST INSTANCE
    i = int(sys.argv[1])
    L = int(sys.argv[2])
    R = int(sys.argv[3])
    T = int(sys.argv[4])

    network_type = "LGT"
    file_name = f"TEST[{network_type}_L{L}_R{R}_T{T}]_FeatImpHeur"

    run_main(i, L, R, T, network_type, full_leaf_set=True, progress=False, file_name=file_name)
