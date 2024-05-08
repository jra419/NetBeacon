import re
import numpy as np
from itertools import product
from .utils import list_to_proba, get_model_table_range_mark, get_value_mask


# Get all the thresholds that appear in the tree
# model_file: tree model file
# keys: the list of features
# tree_num: the number of trees
def get_rf_feature_thres(model_file, keys, tree_num):
    feat_dict = {}

    for key in keys:
        feat_dict[key] = []

    for i in range(tree_num):
        with open(model_file + '_{}.dot'.format(i), 'r') as f:
            lines = f.readlines()

        for line in lines:
            if "[" in line:
                m = re.search(r".*\[label=\"(.*?) <= (.*?)\\n.*", line.strip(), re.M | re.I)
                if m:
                    feat_dict[m.group(1)].append(float(m.group(2)))

    for key in feat_dict.keys():
        for i in range(len(feat_dict[key])):
            # Rounding down, then adding 1, because the node in rf is f <= a
            feat_dict[key][i] = int(feat_dict[key][i]) + 1
        feat_dict[key] = list(np.unique(np.array(feat_dict[key])))

    return feat_dict

# Get the model table table entries
# model_file: model file
# feat_dict: the thresholds of each feature
# key_encode_bits: range mark
# tree_num: the number of trees in the forest
# pkts: the first few packets, optional
# The return value is a list, each element represents a table item
# the content is the range mark of each feature and the classification result
def get_rf_trees_table_entries(model_file, keys, feat_dict, key_encode_bits, tree_num, pkts=None):
    # Each row is a leaf node, recording the smallest threshold index in the left subtree
    # and the smallest threshold index (negative) in the right subtree on the path of that node
    tree_data = []
    tree_leaves= []
    trees = []
    leaf_index = []
    leaf_info = []
    trees.append(len(tree_leaves))

    for i in range(tree_num):
        with open(model_file+'_{}.dot'.format(i), 'r') as f:
            lines = f.readlines()
        nodes = {}
        for j in range(len(lines)):
            line = lines[j]

            if "label=\"" in line and "->" not in line:
                if "[label=\"gini" in line or "[label=\"entropy" in line:
                    m = re.search(r"(.*?) \[label=.*value = (.*?)\\nclass.*",
                                  line.strip(), re.M|re.I)
                    nodes[m.group(1)] = {}
                    # Assumption that there are no more than 1000 different feature thresholds.
                    nodes[m.group(1)]['path'] = [1000, 0] * len(keys)
                    leaf_info.append(list_to_proba(m.group(2)))
                    leaf_index.append(int(m.group(1)))
                else:
                    m = re.search(r"(.*?) \[label=\"(.*?) <= (.*?)\\n.*",
                                  line.strip(), re.M|re.I)
                    nodes[m.group(1)] = {}
                    nodes[m.group(1)]['info'] = [m.group(2),m.group(3)] #feat and threshold
                    nodes[m.group(1)]['path'] = [1000, 0] * len(keys)
                    nodes[m.group(1)]['have_left'] = False

            if "->" in line:
                m = re.search(r"(.*?) -> (.*?) ", line.strip(), re.M|re.I)
                [feat, thres] = nodes[m.group(1)]['info']
                thres = int(float(thres)) + 1
                nodes[m.group(2)]['path'] = nodes[m.group(1)]['path'].copy()
                # Left subtree
                if not nodes[m.group(1)]['have_left']:
                    nodes[m.group(2)]['path'][keys.index(feat) * 2] = \
                        min(nodes[m.group(2)]['path'][keys.index(feat) * 2],
                            feat_dict[feat].index(thres) + 1)
                    nodes[m.group(1)]['have_left'] = True
                else:
                    nodes[m.group(2)]['path'][keys.index(feat)* 2 + 1] = \
                        min(nodes[m.group(2)]['path'][keys.index(feat) * 2 + 1],
                            -feat_dict[feat].index(thres) - 1)
                # Leaf node
                if 'have_left' not in nodes[m.group(2)].keys():
                    tree_leaves.append(nodes[m.group(2)]['path'])
        trees.append(len(tree_leaves))

    print("trees: ",trees,len(leaf_info))
    print("judge leaf conflict ...")

    loop_val = []
    for i in range(len(trees))[:-1]:
        loop_val.append(range(trees[i],trees[i+1]))

    print(loop_val)

    for tup in product(*loop_val):
        flag = 0

        for f in range(len(keys)): #Check for conflicting feature values
            a = 1000
            b = 1000

            for i in tup:
                a = min(tree_leaves[i][f*2], a)
                b = min(tree_leaves[i][f*2+1], b)

            if a + b <= 0:
                flag = 1
                break

        # Semantic conflict check can be added here
        if flag == 0:
            #print("-- ", tup, sigmoid(leafs[i] + leafs[j]))
            if pkts is None:
                tree_data.append([]) #
            else:
                tree_data.append([pkts])

            for f in range(len(keys)):
                a = 1000
                b = 1000

                for i in tup:
                    a = min(tree_leaves[i][f*2], a)
                    b = min(tree_leaves[i][f*2+1], b)

                key = keys[f]
                te = get_model_table_range_mark(key_encode_bits[key], a, b, len(feat_dict[key]))
                # The value and mask of each feature
                tree_data[-1].extend([int(get_value_mask(te, key_encode_bits[key])[0], 2),
                                      int(get_value_mask(te, key_encode_bits[key])[1], 2)])

            leaf_sum = leaf_info[tup[0]].copy()
            for i in tup[1:]:
                for j in range(len(leaf_sum)):
                    leaf_sum[j] += leaf_info[i][j]

            # Classification probabilities list
            tree_data[-1].append(np.array(leaf_sum) / len(tup))
            #print(tup, np.max(leaf_sum) / len(tup))

    return tree_data
