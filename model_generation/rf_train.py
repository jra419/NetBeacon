import sys
import argparse
import json
import statistics
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz


def fc_process(train_file, inf_point):
    print(f'Processing the input json with inference point: {inf_point}')

    fc_flows = []

    with open(train_file, 'r') as tf:
        json_data = json.load(tf)

        for data in json_data:
            pkt_num = data['packet_num']
            if pkt_num < inf_point:
                continue

            len_seq = data['len_seq'][:inf_point]
            ts_seq = data['ts_seq'][:inf_point]

            pkt_size_max = max(len_seq)
            pkt_size_min = min(len_seq)
            pkt_size_mean = sum(len_seq)/inf_point
            pkt_size_variance = statistics.variance(len_seq)

            flow_iat_min = sys.maxsize
            for i in range(len(ts_seq)-1):
                if flow_iat_min > ts_seq[i+1] - ts_seq[i]:
                    flow_iat_min = ts_seq[i+1] - ts_seq[i]

            fc_flows.append([data['source'], pkt_size_max, pkt_size_min, pkt_size_mean,
                             pkt_size_variance, flow_iat_min, str(data['label'])])

    df_flows = pd.DataFrame(fc_flows, columns = ['source', 'pkt_size_max', 'pkt_size_min',
                                                    'pkt_size_mean', 'pkt_size_variance',
                                                    'flow_iat_min', 'label'])

    return df_flows

def train(df_flows, inf_point, export_graph):
    print(f'RF model training with inference point: {inf_point}')

    df_features = df_flows.iloc[:, [1, 2, 3, 4, 5]]
    rfc = RandomForestClassifier(n_estimators=3, max_depth=7)
    rfc.fit(df_features, df_flows['label'])

    if export_graph:
        for i in range(len(rfc.estimators_)):
            export_graphviz(rfc.estimators_[i],
                            out_file='rf_tree_' + inf_point + '_{}.dot'.format(i),
                            feature_names = df_features.columns,
                            class_names = df_flows['label'].values,
                            rounded = True, proportion = False, precision = 4, filled = True)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='RF training.')
    argparser.add_argument('--train_file', type=str, help='Train file path')
    argparser.add_argument('--graph', type=bool, default=False, help='Export RF model as dot graph')
    args = argparser.parse_args()

    # Process the input json for the various inference points.
    df_flows_8      = fc_process(args.train_file, 8)
    df_flows_32     = fc_process(args.train_file, 32)
    df_flows_256    = fc_process(args.train_file, 256)
    df_flows_512    = fc_process(args.train_file, 512)
    df_flows_2048   = fc_process(args.train_file, 2048)

    # Training: RF.
    train(df_flows_8, '8', args.graph)
    train(df_flows_32, '32', args.graph)
    train(df_flows_256, '256', args.graph)
    train(df_flows_512, '512', args.graph)
    train(df_flows_2048, '2048', args.graph)
