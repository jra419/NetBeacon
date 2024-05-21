import os
import sys
import argparse
import json
import statistics
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz


def fc_pkt(train_file):
    print('Processing input features (packet-level).')

    fc_pkts = []

    with open(train_file, 'r') as tf:
        json_data = json.load(tf)

        for data in json_data:
            pkt_num = data['packet_num']
            proto = data['proto']

            for pkt in range(pkt_num):
                pkt_len = data['len_seq'][pkt]
                pkt_diffserv = data['ip_tos_seq'][pkt]
                pkt_ttl = data['ip_ttl_seq'][pkt]
                pkt_tcp_off = data['tcp_off_seq'][pkt]
                pkt_tcp_win = data['tcp_win_seq'][pkt]
                pkt_udp_len = data['udp_len_seq'][pkt]

                fc_pkts.append([data['source'], pkt, proto, pkt_len, pkt_diffserv, pkt_ttl,
                                pkt_tcp_off, pkt_tcp_win, pkt_udp_len, str(data['label'])])

    df_pkts = pd.DataFrame(fc_pkts, columns = ['source', 'pkt_index', 'protocol', 'total_len',
                                               'diffserv', 'ttl', 'tcp_offset', 'tcp_win',
                                               'udp_len', 'label'])

    return df_pkts

def fc_flow(train_file, inf_point):
    print(f'Processing input features (flow-level). Inference point: {inf_point}')

    fc_flows = []

    with open(train_file, 'r') as tf:
        json_data = json.load(tf)

        for idx, data in enumerate(json_data):
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
                                                'pkt_size_mean', 'pkt_size_variance', 'flow_iat_min',
                                                'label'])

    return df_flows

def train_pkts(df_pkts, export_graph, train_name):
    print('Packet-level: RF model training')

    df_features = df_pkts.iloc[:, [2, 3, 4, 5, 6, 7, 8]]
    rfc = RandomForestClassifier(n_estimators=3, max_depth=7)
    rfc.fit(df_features, df_pkts['label'])

    outdir = str(Path(__file__).parents[0]) + '/models/' + train_name
    if not os.path.exists(str(Path(__file__).parents[0]) + '/models/' + train_name):
        os.mkdir(outdir)

    if export_graph:
        for i in range(len(rfc.estimators_)):
            export_graphviz(rfc.estimators_[i],
                            out_file = outdir + '/rf_tree_pkt_{}.dot'.format(i),
                            feature_names = df_features.columns,
                            class_names = df_pkts['label'].values,
                            rounded = True, proportion = False, precision = 4, filled = True)

def train_flows(df_flows, inf_point, export_graph, train_name):
    print(f'Flow-level: RF model training with inference point: {inf_point}')

    df_features = df_flows.iloc[:, [1, 2, 3, 4, 5]]
    rfc = RandomForestClassifier(n_estimators=3, max_depth=7)
    rfc.fit(df_features, df_flows['label'])

    outdir = str(Path(__file__).parents[0]) + '/models/' + train_name
    if not os.path.exists(str(Path(__file__).parents[0]) + '/models/' + train_name):
        os.mkdir(outdir)

    if export_graph:
        for i in range(len(rfc.estimators_)):
            export_graphviz(rfc.estimators_[i],
                            out_file = outdir + '/rf_tree_flow_' + inf_point + '_{}.dot'.format(i),
                            feature_names = df_features.columns,
                            class_names = df_flows['label'].values,
                            rounded = True, proportion = False, precision = 4, filled = True)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='RF training.')
    argparser.add_argument('--train_file', type=str, help='Train file path')
    argparser.add_argument('--train_name', type=str, help='Dataset/Trace name')
    argparser.add_argument('--graph', type=bool, default=False, help='Export RF model as dot graph')
    args = argparser.parse_args()

    # Pkt-level: Process the input json.
    df_pkts         = fc_pkt(args.train_file)

    # Flow-level: Process the input json for the various inference points.
    df_flows_8      = fc_flow(args.train_file, 8)
    df_flows_32     = fc_flow(args.train_file, 32)
    df_flows_256    = fc_flow(args.train_file, 256)
    df_flows_512    = fc_flow(args.train_file, 512)
    df_flows_2048   = fc_flow(args.train_file, 2048)

    # Training: RF.
    train_pkts(df_pkts, args.graph, args.train_name)
    train_flows(df_flows_8, '8', args.graph, args.train_name)
    train_flows(df_flows_32, '32', args.graph, args.train_name)
    train_flows(df_flows_256, '256', args.graph, args.train_name)
    train_flows(df_flows_512, '512', args.graph, args.train_name)
    train_flows(df_flows_2048, '2048', args.graph, args.train_name)
