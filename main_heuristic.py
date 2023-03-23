from datetime import datetime
import pandas as pd
import numpy as np
import copy
import sys
import csv
import warnings

import CPH as CPH

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


def run_heuristic(tree_set=None, tree_set_newick=None, inst_num=0, repeats=1, time_limit=None,
                  progress=False,  pick_triv=False, pick_ml=False, pick_ml_triv=False,
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


def run_main(i, l, network_type, ret=None, forest_size=None, feat_imp_heur=False,
             repeats=1000, time_limit=None, ml_name=None, full_leaf_set=True, ml_thresh=None, progress=False, file_name=None,
             missing_leaves=0):
    if network_type == "normal":
        file_info = f"L{l}_R{ret}_normal"
        test_inst_map = "normal"
    elif network_type == "ZODS":
        file_info = f"L{l}_R{ret}_T{forest_size}_ZODS"
        test_inst_map = "ZODS"
    else:
        file_info = f"L{l}_R{ret}_T{forest_size}_LGT"
        test_inst_map = "LGT"

    if missing_leaves > 0:
        file_info += f"_PART[{missing_leaves}]"

    # ML MODEL
    model_name = f"data/RFModels/rf_cherries_{ml_name}.joblib"

    # save results
    if feat_imp_heur:
        columns = ["ML", "TrivialML", "Rand", "TrivialRand", "FeatImp", "UB"]
    else:
        columns = ["ML", "TrivialML", "Rand", "TrivialRand", "UB"]

    score = pd.DataFrame(
        index=pd.MultiIndex.from_product([[i], ["RetNum", "Time"], np.arange(repeats)]),
        columns=columns, dtype=float)

    df_seq = pd.DataFrame()
    # INSTANCE
    tree_set_newick = f"data/test/{test_inst_map}/newick/tree_set_newick_{file_info}_{i}.txt"

    # ML HEURISTIC
    ret_score, time_score, seq_ml, df_pred = run_heuristic(
        tree_set_newick=tree_set_newick,
        inst_num=i,
        repeats=1,
        time_limit=time_limit,
        pick_ml=True,
        relabel=True,
        model_name=model_name,
        problem_type="ML",
        full_leaf_set=full_leaf_set,
        ml_thresh=ml_thresh,
        progress=progress)
    score.loc[i, "RetNum", 0]["ML"] = copy.copy(ret_score[0])
    score.loc[i, "Time", 0]["ML"] = copy.copy(time_score[0])
    ml_time = score.loc[i, "Time", 0]["ML"]
    ml_ret = int(score.loc[i, "RetNum"]["ML"][0])
    df_seq = pd.concat([df_seq, pd.Series(seq_ml)], axis=1)

    # ML Trivial HEURISTIC
    ret_score, time_score, seq_ml_triv = run_heuristic(
        tree_set_newick=tree_set_newick,
        inst_num=i,
        repeats=1,
        time_limit=time_limit,
        pick_ml_triv=True,
        relabel=True,
        model_name=model_name,
        problem_type="TrivialML",
        full_leaf_set=full_leaf_set,
        ml_thresh=ml_thresh,
        progress=progress)

    score.loc[i, "RetNum", 0]["TrivialML"] = copy.copy(ret_score[0])
    score.loc[i, "Time", 0]["TrivialML"] = copy.copy(time_score[0])
    ml_triv_ret = int(score.loc[i, "RetNum"]["TrivialML"][0])
    df_seq = pd.concat([df_seq, pd.Series(seq_ml_triv)], axis=1)

    # RANDOM HEURISTIC
    ret_score, time_score, seq_ra = run_heuristic(
        tree_set_newick=tree_set_newick,
        inst_num=i,
        repeats=repeats,
        time_limit=ml_time,
        problem_type="Rand",
        pick_random=True,
        relabel=False,
        full_leaf_set=full_leaf_set,
        progress=progress)

    for r, ret in ret_score.items():
        score.loc[i, "RetNum", r]["Rand"] = copy.copy(ret)
        score.loc[i, "Time", r]["Rand"] = copy.copy(time_score[r])
    ra_ret = int(min(score.loc[i, "RetNum"]["Rand"]))
    df_seq = pd.concat([df_seq, pd.Series(seq_ra)], axis=1)

    # TRIVIAL RANDOM
    ret_score, time_score, seq_tr = run_heuristic(
        tree_set_newick=tree_set_newick,
        inst_num=i,
        repeats=repeats,
        time_limit=ml_time,
        pick_triv=True,
        relabel=True,
        problem_type="TrivialRand",
        full_leaf_set=full_leaf_set,
        progress=progress)

    for r, ret in ret_score.items():
        score.loc[i, "RetNum", r]["TrivialRand"] = copy.copy(ret)
        score.loc[i, "Time", r]["TrivialRand"] = copy.copy(time_score[r])
    tr_ret = int(min(score.loc[i, "RetNum"]["TrivialRand"]))
    df_seq = pd.concat([df_seq, pd.Series(seq_tr)], axis=1)

    if feat_imp_heur:
        # FEATURE IMPORTANCE
        ret_score, time_score, seq_fi = run_heuristic(
            tree_set_newick=tree_set_newick,
            inst_num=i,
            repeats=1,
            time_limit=ml_time,
            pick_feat_imp=True,
            relabel=True,
            problem_type="FeatImp",
            full_leaf_set=full_leaf_set,
            progress=progress)

        for r, ret in ret_score.items():
            score.loc[i, "RetNum", r]["FeatImp"] = copy.copy(ret)
            score.loc[i, "Time", r]["FeatImp"] = copy.copy(time_score[r])
        fi_ret = int(min(score.loc[i, "RetNum"]["FeatImp"]))
        df_seq = pd.concat([df_seq, pd.Series(seq_fi)], axis=1)
        time_FeatImp = np.round(time_score[0], 2)
    else:
        fi_ret = None
        time_FeatImp = None

    # upper bound of ret
    idx = pd.IndexSlice
    env_info_file = f"data/test/{test_inst_map}/instances/tree_data_{file_info}_{i}.pickle"
    env_info = pd.read_pickle(env_info_file)
    ub_ret = int(env_info["metadata"]["rets"])
    score.loc[idx[i, "RetNum", :], "UB"] = ub_ret

    # print results
    if progress:
        mean_Rand = np.round(score.loc[:, "RetNum", :]["Rand"].mean(), 2)
        quant_Rand = np.round(score.loc[:, "RetNum", :]["Rand"].quantile(0.9), 2)
        mean_TrivialRand = np.round(score.loc[:, "RetNum", :]["TrivialRand"].mean(), 2)
        quant_TrivialRand = np.round(score.loc[:, "RetNum", :]["TrivialRand"].quantile(0.9), 2)
        print()
        print("FINAL RESULTS\n"
              f"Instance = {i} \n"
              "RETICULATIONS\n"
              f"ML            = {ml_ret}\n"
              f"TrivialML     = {ml_triv_ret}\n"
              f"Rand          = [{ra_ret}, {mean_Rand}, {quant_Rand}]\n"
              f"TrivialRand   = [{tr_ret}, {mean_TrivialRand}, {quant_TrivialRand}]\n"
              f"FeatImp       = {fi_ret}\n"
              f"Reference     = {ub_ret}\n"
              f"ML time       = {np.round(ml_time, 2)}s\n"
              f"FeatImp time  = {time_FeatImp}s")
    else:
        if feat_imp_heur:
            print(i, ml_ret, ml_triv_ret, ra_ret, tr_ret, fi_ret, ub_ret, ml_time)
        else:
            print(i, ml_ret, ml_triv_ret, ra_ret, tr_ret, ub_ret, ml_time)

    # SAVE DATAFRAMES
    # scores
    score.dropna(axis=0, how="all").to_pickle(f"data/results/inst_results/heuristic_scores_{file_name}_{i}.pickle")
    # ml predictions
    df_pred.to_pickle(f"data/results/inst_results/cherry_prediction_info_{file_name}_{i}.pickle")
    # best sequences
    df_seq.columns = score.columns[:-1]
    df_seq.index = pd.MultiIndex.from_product([[i], df_seq.index])
    df_seq.to_pickle(f"data/results/inst_results/cherry_seq_{file_name}_{i}.pickle")


# MAIN
if __name__ == "__main__":
    # TEST INSTANCE
    i = int(sys.argv[1])
    L = int(sys.argv[2])
    R = int(sys.argv[3])
    T = int(sys.argv[4])
    normal = int(sys.argv[5])
    zods = int(sys.argv[6])
    missing_leaves = int(sys.argv[7])

    if normal:
        ret = R
        forest_size = None
        network_type = "normal"
    else:
        ret = R
        forest_size = T
        if zods:
            network_type = "ZODS"
        else:
            network_type = "LGT"

    # ML MODEL
    maxL = int(sys.argv[8])
    net_num = int(sys.argv[9])
    normal_ML_trained = int(sys.argv[10])

    if normal_ML_trained:
        ML_network_type = "normal"
    else:
        ML_network_type = "LGT"

    ml_name = f"N{net_num}_maxL{maxL}_{ML_network_type}_balanced"
    if len(sys.argv) == 12:
        ml_thresh = int(sys.argv[11])
    else:
        ml_thresh = None

    # FILE NAME
    if normal:
        file_name = f"TEST[normal_L{L}_R{ret}]_ML[{ML_network_type}_N{net_num}_maxL{maxL}]"
    elif zods:
        file_name = f"TEST[ZODS_L{L}_R{ret}_T{forest_size}]_ML[{ML_network_type}_N{net_num}_maxL{maxL}]"
    else:
        file_name = f"TEST[LGT_L{L}_R{ret}_T{forest_size}]_ML[{ML_network_type}_N{net_num}_maxL{maxL}]"

    if missing_leaves > 0:
        file_name += f"_PART[{missing_leaves}]"
        full_leaf_set = False
    else:
        file_name += "_all"
        full_leaf_set = True

    run_main(i, L, network_type, ret, forest_size, ml_name=ml_name, full_leaf_set=full_leaf_set, ml_thresh=None,
             progress=False, file_name=file_name, feat_imp_heur=False)

