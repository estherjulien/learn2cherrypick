from datetime import datetime
import pandas as pd
import numpy as np
import argparse
import copy
import csv
import warnings
import os


import CPH as CPH

warnings.simplefilter(action='ignore', category=FutureWarning)

'''
Code consisting of main run file and two functions:
- run_heuristic:            1. open tree set and make CPH.PhT environment for each tree
                            2. run cherry picking heuristic (CPH)
                            3. return results
- run_main:                 run CPH with:
                                1. Rand
                                2. TrivialRand

RUN in terminal:
'''


def run_heuristic(tree_set=None, tree_set_newick=None, inst_num=0, repeats=1, time_limit=None,
                  progress=False, pick_triv=False, pick_ml=False, pick_ml_triv=False,
                  pick_random=False, pick_feat_imp=False, model_name=None, relabel=False, problem_type="",
                  full_leaf_set=True, ml_thresh=None):
    # READ TREE SET
    now = datetime.now().time()
    if progress:
        print(f"Instance {inst_num} {problem_type}: Start at {now}")

    if tree_set is None and tree_set_newick is not None:
        # Empty set of inputs
        inputs = []

        # Read each line of the input file with name set by "option_file_argument"
        f = open(tree_set_newick, "rt")
        reader = csv.reader(f, delimiter='~', quotechar='|')
        for row in reader:
            new_line = str(row[0])
            if new_line[-1] == ";":
                inputs.append(new_line[:-1])
            else:
                inputs.append(new_line)
        f.close()

        # Make the set of inputs usable for all algorithms: use the CPH class
        tree_set = CPH.Input_Set(newick_strings=inputs, instance=inst_num)

    # RUN HEURISTIC CHERRY PICKING SEQUENCE
    # Run the heuristic to find a cherry-picking sequence `seq' for the set of input trees.
    # Arguments are set as given by the terminal arguments
    seq_dist, seq, df_pred = tree_set.CPSBound(repeats=repeats,
                                               progress=progress,
                                               time_limit=time_limit,
                                               pick_triv=pick_triv,
                                               pick_ml=pick_ml,
                                               pick_ml_triv=pick_ml_triv,
                                               pick_random=pick_random,
                                               pick_feat_imp=pick_feat_imp,
                                               relabel=relabel,
                                               model_name=model_name,
                                               ml_thresh=ml_thresh,
                                               problem_type=problem_type,
                                               full_leaf_set=full_leaf_set)

    # Output the computation time for the heuristic
    now = datetime.now().time()
    if progress:
        print(f"Instance {inst_num} {problem_type}: Finish at {now}")
        print(f"Instance {inst_num} {problem_type}: Computation time heuristic: {tree_set.CPS_Compute_Time}")
        print(f"Instance {inst_num} {problem_type}: Reticulation number = {min(tree_set.RetPerTrial.values())}")
    if pick_ml:
        return tree_set.RetPerTrial, tree_set.DurationPerTrial, seq, df_pred
    else:
        return tree_set.RetPerTrial, tree_set.DurationPerTrial, seq


def run_main(file_name_newick, i, repeats=1000, time_limit=None, progress=False,
             out_file_retics="", out_file_seqs=""):

    if not out_file_retics:
        out_file_retics = f"data/results/inst_results/cherry_retics_{i}.pkl"
    if not out_file_seqs:
        out_file_seqs = f"data/results/inst_results/cherry_seqs_{i}.pkl"

    # save results
    columns = ["Rand", "TrivialRand"]

    score = pd.DataFrame(
        index=pd.MultiIndex.from_product([[i], ["RetNum", "Time"], np.arange(repeats)]),
        columns=columns, dtype=float)

    df_seq = pd.DataFrame()
    # INSTANCE

    # RANDOM HEURISTIC
    ret_score, time_score, seq_ra = run_heuristic(
        tree_set_newick=file_name_newick,
        inst_num=i,
        repeats=repeats,
        time_limit=time_limit,
        problem_type="Rand",
        pick_random=True,
        relabel=True,
        full_leaf_set=False,
        progress=False)

    for r, ret in ret_score.items():
        score.loc[i, "RetNum", r]["Rand"] = copy.copy(ret)
        score.loc[i, "Time", r]["Rand"] = copy.copy(time_score[r])
    ra_ret = int(min(score.loc[i, "RetNum"]["Rand"]))
    df_seq = pd.concat([df_seq, pd.Series(seq_ra)], axis=1)

    # TRIVIAL RANDOM
    ret_score, time_score, seq_tr = run_heuristic(
        tree_set_newick=file_name_newick,
        inst_num=i,
        repeats=repeats,
        time_limit=time_limit,
        pick_triv=True,
        relabel=True,
        problem_type="TrivialRand",
        full_leaf_set=False,
        progress=False)

    for r, ret in ret_score.items():
        score.loc[i, "RetNum", r]["TrivialRand"] = copy.copy(ret)
        score.loc[i, "Time", r]["TrivialRand"] = copy.copy(time_score[r])
    tr_ret = int(min(score.loc[i, "RetNum"]["TrivialRand"]))
    df_seq = pd.concat([df_seq, pd.Series(seq_tr)], axis=1)

    # print results
    if progress:
        print()
        print("FINAL RESULTS\n"
              f"Instance = {i} \n"
              "RETICULATIONS\n"
              f"Rand          = {ra_ret}\n"
              f"TrivialRand   = {tr_ret}\n"
              )
    else:
        print(i, ra_ret, tr_ret)

    # SAVE DATAFRAMES
    os.makedirs("data/results/inst_results", exist_ok=True)
    # scores
    score.dropna(axis=0, how="all").to_pickle(out_file_retics)
    # best sequences
    df_seq.columns = score.columns
    df_seq.index = pd.MultiIndex.from_product([[i], df_seq.index])
    df_seq.to_pickle(out_file_seqs)


# MAIN
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trees_file", type=str)
    parser.add_argument("--inst_name", type=str, default="trees")

    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--time_limit", type=int, default=None)

    parser.add_argument("--out_file_retics", type=str, default="")
    parser.add_argument("--out_file_seqs", type=str, default="")
    args = parser.parse_args()

    run_main(args.trees_file, args.inst_name, args.repeats, args.time_limit,
             out_file_retics=args.out_file_retics, out_file_seqs=args.out_file_seqs)



