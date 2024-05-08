import os
import pickle
from tree_to_table.rf import get_rf_feature_thres, get_rf_trees_table_entries
from tree_to_table.xgb import get_xgb_feature_thres, get_xgb_trees_table_entries
from tree_to_table.utils import get_feature_table_entries, get_bin_table


current_path = os.getcwd()
dir_path = os.path.abspath(os.path.dirname(os.getcwd()))

## Merge flowsize prediction and packet level classification, range mark of both can be shared
def get_flow_size_and_class_pkt():
    pkt_feat = ['proto', 'total_len', 'diffserv', 'ttl',
                'tcp_dataOffset', 'tcp_window', 'udp_length']
    pkt_feat_bits = [8, 16, 8, 8, 4, 16, 16]
    key_bits = {}
    for i in range(len(pkt_feat)):
        key_bits[pkt_feat[i]] = pkt_feat_bits[i]

    # Flow size prediction
    flow_size_model_file = current_path+'/models/flow_size_predict_1xgb.txt'
    # Packet level classification
    class_pkt_model_file = current_path+'/models/class_pkt_1rf'

    feat_dict_flow_size = get_xgb_feature_thres(flow_size_model_file, pkt_feat)

    class_pkt_tree_num = 1
    feat_dict_class_pkt = get_rf_feature_thres(class_pkt_model_file, pkt_feat, class_pkt_tree_num)

    feat_dict = {}
    for key in pkt_feat:
        feat_dict[key] = list(set(feat_dict_flow_size[key]) | set(feat_dict_class_pkt[key]))
        feat_dict[key].sort()
        print(key, len(feat_dict_flow_size[key]), feat_dict_flow_size[key])
        print(key, len(feat_dict_class_pkt[key]), feat_dict_class_pkt[key])
        print('--', key, len(feat_dict[key]), feat_dict[key])

    #range mark bits, maximum value of the number of thresholds in each model, respectively
    pkt_feat_mark_bit = [8, 144, 24, 64, 8, 48, 72]
    range_mark_bits = {}
    for i in range(len(pkt_feat)):
        range_mark_bits[pkt_feat[i]] = pkt_feat_mark_bit[i]

    feat_table_data = get_feature_table_entries(feat_dict,key_bits, range_mark_bits)
    sum_e = 0
    for key in feat_table_data.keys():
        sum_e += len(feat_table_data[key])
    print(f'Feature table entries: {sum_e}')

    tree_data_flow_size = get_xgb_trees_table_entries(flow_size_model_file, pkt_feat,
                                                      feat_dict, range_mark_bits)
    print(f'Flow size tree table entries: {len(tree_data_flow_size)}')

    tree_data_class_pkt = get_rf_trees_table_entries(class_pkt_model_file, pkt_feat, feat_dict,
                                                     range_mark_bits, class_pkt_tree_num)
    print(f'Class pkt tree table entries: {len(tree_data_class_pkt)}')

    with open(dir_path + '/switch/control_plane/flow_size_and_class_pkt.pkl', 'wb') as f:
        pickle.dump([feat_table_data, tree_data_flow_size, tree_data_class_pkt], f, protocol=2)

## Multi-phase classification
def get_class_flow():
    pkt_flow_feat = ['pkt_size_max',            # Max pkt size
                     'flow_iat_min',            # Min ipd
                     'bin_5',                   # Pkt num whose size in [80,96)
                     'bin_3',                   # Pkt num whose size in [48, 64)
                     'pkt_size_var_approx',     # Pkt size variance
                     'pkt_size_avg',            # Pkt size mean
                     'pkt_size_min']            # Min pkt size
    pkt_flow_feat_bit = [16, 32, 8, 16, 32, 32, 16]

    # Convert the aggregate features
    # 16: packet size bits
    bin_table = get_bin_table(pkt_flow_feat, 16)

    class_flow_model_files = [
            current_path + '/models/class_flow_phase_2pkt_1rf',
            current_path + '/models/class_flow_phase_4pkt_1rf',
            current_path + '/models/class_flow_phase_8pkt_1rf',
            current_path + '/models/class_flow_phase_32pkt_1rf',
            current_path + '/models/class_flow_phase_256pkt_1rf',
            current_path + '/models/class_flow_phase_512pkt_1rf',
            current_path + '/models/class_flow_phase_2048pkt_1rf',]
    class_flow_tree_nums = [1, 1, 1, 1, 1, 1, 1]

    max_feat_thres = {}
    for i in range(len(pkt_flow_feat)):
        max_feat_thres[pkt_flow_feat[i]] = 0

    count = 0
    feat_dicts = []
    for model_file in class_flow_model_files:
        tree_num = class_flow_tree_nums[count]
        feat_dict = get_rf_feature_thres(model_file, pkt_flow_feat, tree_num)
        for key in feat_dict.keys():
            print(key, len(feat_dict[key]), feat_dict[key])
            if max_feat_thres[key] < len(feat_dict[key]):
                max_feat_thres[key] = len(feat_dict[key])
        count += 1
        feat_dicts.append(feat_dict)

    print(f'Max feat thres: {max_feat_thres}')

    # Max feat thres
    pkt_flow_mark_bit = [80, 72, 24, 32, 80, 72, 48]
    feat_key_bits = {}
    range_mark_bits = {}
    for i in range(len(pkt_flow_feat)):
        feat_key_bits[pkt_flow_feat[i]] = pkt_flow_feat_bit[i]
        range_mark_bits[pkt_flow_feat[i]] = pkt_flow_mark_bit[i]

    # Model phases
    pkt_flow_model_pkts = [2, 4, 8, 32, 256, 512, 2048]
    feat_table_data_all = {}
    for i in range(len(pkt_flow_feat)):
        feat_table_data_all[pkt_flow_feat[i]] = []
    tree_data_all = []

    for i in range(len(class_flow_model_files)):
        tree_num = class_flow_tree_nums[i]
        model_file = class_flow_model_files[i]
        pkts = pkt_flow_model_pkts[i]
        feat_dict = feat_dicts[i]
        feat_table_datas = get_feature_table_entries(feat_dict, feat_key_bits,
                                                     range_mark_bits, pkts=pkts)
        sum_e = 0
        for key in feat_table_datas.keys():
            sum_e += len(feat_table_datas[key])
        print(f'Feature table entries: {sum_e}')

        tree_data = get_rf_trees_table_entries(model_file, pkt_flow_feat, feat_dict,
                                               range_mark_bits, tree_num, pkts=pkts)

        print(f'Tree table entries: {len(tree_data)}')
        print(f'All table entries: {sum_e + len(tree_data)}')

        for i in range(len(pkt_flow_feat)):
            feat_table_data_all[pkt_flow_feat[i]].extend(feat_table_datas[pkt_flow_feat[i]])

        tree_data_all.extend(tree_data)

    for key in pkt_flow_feat:
        print(key, len(feat_table_data_all[key]))
    print(len(tree_data_all))

    with open(dir_path + '/switch/control_plane/bin_table_and_class_flow.pkl', 'wb') as f:
        pickle.dump([bin_table, feat_table_data_all, tree_data_all], f, protocol=2)

get_flow_size_and_class_pkt()
get_class_flow()
