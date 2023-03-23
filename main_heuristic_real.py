import time

from main_heuristic import run_heuristic
import pandas as pd
import numpy as np
import copy
import sys
import warnings


def run_main(i, l=None, forest_size=None, repeats=1000, time_limit=None, ml_name=None, full_leaf_set=True,
             ml_thresh=None, progress=False, file_name=None, feat_imp_heur=False):
    if l == 10:
        tot_trees = 770
    elif l == 20:
        tot_trees = 1684
    elif l == 30:
        tot_trees = 815
    elif l == 40:
        tot_trees = 386
    elif l == 50:
        tot_trees = 290
    elif l == 60:
        tot_trees = 205
    elif l == 80:
        tot_trees = 105
    elif l == 100:
        tot_trees = 53
    elif l == 150:
        tot_trees = 21
    else:
        return None

    # ML MODEL
    model_name = f"AMBCode/Data/RFModelsLGT/rf_cherries_{ml_name}.joblib"
    # save results
    if feat_imp_heur:
        columns = ["ML", "TrivialML", "Rand", "TrivialRand", "FeatImp", "UB"]
    else:
        columns = ["ML", "TrivialML", "Rand", "TrivialRand", "UB"]
    df_seq = pd.DataFrame()
    score = pd.DataFrame(
        index=pd.MultiIndex.from_product([[i], ["RetNum", "Time"], np.arange(repeats)]),
        columns=columns, dtype=float)
    # INSTANCE
    tree_set_newick = f"data/test/real/compare_newick/{l}_leaves_{tot_trees}_trees_{forest_size}_trees_{i}.txt"

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
              f"ML time       = {np.round(ml_time, 2)}s\n"
              f"FeatImp time  = {time_FeatImp}s")
    else:
        if feat_imp_heur:
            print(i, ml_ret, ml_triv_ret, ra_ret, tr_ret, fi_ret, ml_time)
        else:
            print(i, ml_ret, ml_triv_ret, ra_ret, tr_ret, ml_time)

    # SAVE DATAFRAMES
    # scores
    score.dropna(axis=0, how="all").to_pickle(f"data/results/inst_results/heuristic_scores_{file_name}_{i}.pickle")
    # ml predictions
    df_pred.to_pickle(f"data/results/inst_results/cherry_prediction_info_{file_name}_{i}.pickle")
    # best sequences
    df_seq.columns = score.columns[:-1]
    df_seq.index = pd.MultiIndex.from_product([[i], df_seq.index])
    df_seq.to_pickle(f"data/results/inst_results/cherry_seq_{file_name}_{i}.pickle")


if __name__ == "__main__":
    # TEST INSTANCE
    i = int(sys.argv[1])
    L = int(sys.argv[2])
    T = int(sys.argv[3])
    # ML MODEL
    maxL = int(sys.argv[4])
    net_num = int(sys.argv[5])

    ML_network_type = "LGT"

    ml_name = f"N{net_num}_maxL{maxL}_{ML_network_type}_balanced_debugged"
    file_name = f"TEST[REAL_L{L}_T{T}]_ML[{ML_network_type}_N{net_num}_maxL{maxL}]"

    warnings.simplefilter(action='ignore', category=FutureWarning)
    st = time.time()
    run_main(i, l=L, forest_size=T, ml_name=ml_name, progress=False, file_name=file_name)
    print("runtime = ", time.time() - st)
