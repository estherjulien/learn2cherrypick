import numpy as np

'''
Code for transforming networkx Digraph into newick format
'''


def sub_tree_to_newick(G, root=None):
    subgs = []
    for child in G[root]:
        try:
            length = np.round(G.edges[(root, child)]["length"], 3)
        except KeyError:
            length = np.round(G.edges[(root, child)]["lenght"], 3)
        if len(G[child]) > 0:
            subgs.append(sub_tree_to_newick(G, root=child) + f":{length}")
        else:
            subgs.append(str(child) + f":{length}")

    return "(" + ','.join(subgs) + ")"


def tree_to_newick_fun(tree_set, net_num, network_gen="LGT", partial=False, tree_info=""):
    if partial:
        file_name = f"AMBCode/Data/Test/{network_gen}/newick/tree_set_newick{tree_info}_part_{net_num}.txt"
    else:
        file_name = f"AMBCode/Data/Test/{network_gen}/newick/tree_set_newick{tree_info}_{net_num}.txt"

    file = open(file_name, "w+")
    for tree in tree_set.values():
        tree_line = sub_tree_to_newick(tree, 0)
        file.write(tree_line)
        file.write("\n")
    file.close()
