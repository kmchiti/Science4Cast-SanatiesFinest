from scipy import sparse
from datetime import date
import numpy as np
import argparse
import networkx as nx
from tqdm import tqdm
import os
import pickle


def CCPA(SP, CN, num_vertex, alpha=0.8):
    return (alpha * CN) + ((1 - alpha) * (num_vertex / SP))


def args_parser():
    parser = argparse.ArgumentParser(description='Boosting on Graphs')

    parser.add_argument(
        '--dataset',
        type=str,
        default=
        '../SemanticGraph_delta_3_cutoff_0_minedge_1.pkl',
        help=
        'Path to .pkl data (default: ../CompetitionSet2017_3.pkl)'
    )

    parser.add_argument('--bs',
                        type=int,
                        default=256,
                        help='train batch_size (default: 256)')

    parser.add_argument('--epoch',
                        type=int,
                        default=100,
                        help='train batch_size (default: 10)')

    parser.add_argument('--hiddenLayers',
                        type=int,
                        default=100,
                        help='hidden dimension of MLP(default: 100)')

    parser.add_argument('--features',
                        type=str,
                        default='baseline',
                        help='''Features: 

                            baseline: only degrees + their sum!
                            cn: Common Neighbours,
                            shp: shortest_path,
                            pr: page rank,
                            and: average neighbor degree,
                            nd: normalized degree

                        All schemes should be seperated by '_'.
                        (Example: 'aa_ja' will add adamic_adar and jacard.)
                        ''')

    parser.add_argument(
        '--samples',
        type=int,
        default=200000,
        help='Number of samples for the edges! (default: 200k)')

    # GB:
    parser.add_argument('--depth',
                        type=int,
                        default=3,
                        help='Depth of GB (deatult: 3)')

    parser.add_argument('--negRatio',
                        type=float,
                        default=1,
                        help='Negative to positive ratio in dataset (default: 1)')

    parser.add_argument('--loss',
                        type=str,
                        default='deviance',
                        help='Loss of GB (deatult: deviance)')

    parser.add_argument('--criterion',
                        type=str,
                        default='friedman_mse',
                        help='Criterion for GB (deatult: friedman_mse)')

    parser.add_argument('--lr',
                        type=float,
                        default=0.1,
                        help='Learning-rate (default: 0.1)')

    parser.add_argument('--estimators',
                        type=int,
                        default=100,
                        help='number of estimators (default: 100)')

    parser.add_argument('--sampleLeaf',
                        type=int,
                        default=3,
                        help='sample Leaf (default: 3)')

    parser.add_argument('--minSampleSplit',
                        type=int,
                        default=14,
                        help='min Sample Split(default: 16)')

    parser.add_argument('--subsamples',
                        type=float,
                        default=0.8,
                        help='subsample (default: 1.0)')

    args = parser.parse_args()

    return args


def graph_year(fdg, year):

    "returns one slice of the graph (for a specific year)"

    NUM_OF_VERTICES = 64719

    day_origin = date(1990, 1, 1)
    day_curr = date(year, 12, 31)

    all_edges_curr = fdg[fdg[:, 2] < (day_curr - day_origin).days]
    adj_mat_sparse_curr = sparse.csr_matrix(
        (np.ones(len(all_edges_curr)),
         (all_edges_curr[:, 0], all_edges_curr[:, 1])),
        shape=(NUM_OF_VERTICES, NUM_OF_VERTICES))

    print('Number of edges ({}): '.format(year), adj_mat_sparse_curr.getnnz())

    graph = nx.from_scipy_sparse_matrix(adj_mat_sparse_curr)

    return graph


def create_features(args, graph, graph1, graph2, vlist, use_case='train', name=''):

    features_file_name = 'submission_features_{}_4graphs_{}.pickle'.format(args.features, name)

    if use_case == 'submit':
        if os.path.exists(features_file_name):
            with open(features_file_name, 'rb') as handle:
                feature_data = pickle.load(handle)
            return feature_data

    deg = graph.degree()
    deg1 = graph1.degree()
    deg2 = graph2.degree()

    if 'pr' in args.features:
        graph_pr = nx.pagerank(graph)
        graph1_pr = nx.pagerank(graph1)
        graph2_pr = nx.pagerank(graph2)

    if 'and' in args.features:
        graph_and = nx.average_neighbor_degree(graph)
        graph1_and = nx.average_neighbor_degree(graph1)
        graph2_and = nx.average_neighbor_degree(graph2)

    all_features = []
    e = 2 * graph.number_of_edges()
    e1 = 2 * graph1.number_of_edges()
    e2 = 2 * graph2.number_of_edges()
    for i, j in tqdm(vlist, desc="Edge feature computing {}".format(use_case)):

        # Baseline features (degrees)
        features = [
            deg[i],
            deg[j],
            deg[i] + deg[j],
            deg1[i],
            deg1[j],
            deg1[i] + deg1[j],
            deg2[i],
            deg2[j],
            deg2[i] + deg2[j],

            abs(deg1[i] + deg1[j] - (deg[i] + deg[j])),
            abs(deg1[i] + deg1[j] - (deg2[i] + deg2[j])),
            abs(deg[i] + deg[j] - (deg2[i] + deg2[j])),
        ]

        if '_nd' in args.features:
            nd_i = deg[i] / e
            nd_j = deg[j] / e
            nd1_i = deg1[i] / e1
            nd1_j = deg1[j] / e1
            nd2_i = deg2[i] / e2
            nd2_j = deg2[j] / e2

            features.append(nd_i)
            features.append(nd_j)
            features.append(nd_i + nd_j)
            features.append(nd1_i)
            features.append(nd1_j)
            features.append(nd1_i + nd1_j)
            features.append(nd2_i)
            features.append(nd2_j)
            features.append(nd2_i + nd2_j)

            features.append(abs(nd1_i + nd1_j - (nd_i + nd_j)))
            features.append(abs(nd1_i + nd1_j - (nd2_i + nd2_j)))
            features.append(abs(nd_i + nd_j - (nd2_i + nd2_j)))

        # jacard
        if 'ja' in args.features:
            graph_ja = list(nx.jaccard_coefficient(graph, [(i, j)]))[0][2]
            graph1_ja = list(nx.jaccard_coefficient(graph1, [(i, j)]))[0][2]
            graph2_ja = list(nx.jaccard_coefficient(graph2, [(i, j)]))[0][2]
            features.append(graph_ja)
            features.append(graph1_ja)
            features.append(graph2_ja)
            features.append(abs(graph_ja - graph1_ja))
            features.append(abs(graph2_ja - graph1_ja))
            features.append(abs(graph_ja - graph2_ja))

        if 'cn' in args.features:
            path2 = len(list(nx.common_neighbors(graph, i, j)))
            path2_1 = len(list(nx.common_neighbors(graph1, i, j)))
            path2_2 = len(list(nx.common_neighbors(graph2, i, j)))
            features.append(path2)
            features.append(path2_1)
            features.append(path2_2)
            features.append(abs(path2 - path2_1))
            features.append(abs(path2 - path2_2))
            features.append(abs(path2_1 - path2_2))

        if 'shp' in args.features:
            try:
                shortest_path = len(
                    nx.shortest_path(G=graph, source=i, target=j))
            except:
                shortest_path = 100

            try:
                shortest_path_1 = len(
                    nx.shortest_path(G=graph1, source=i, target=j))
            except:
                shortest_path_1 = 100

            try:
                shortest_path_2 = len(
                    nx.shortest_path(G=graph2, source=i, target=j))
            except:
                shortest_path_2 = 100

            features.append(shortest_path)
            features.append(shortest_path_1)
            features.append(shortest_path_2)

            features.append(abs(shortest_path - shortest_path_1))
            features.append(abs(shortest_path_2 - shortest_path_1))
            features.append(abs(shortest_path_2 - shortest_path))

        if 'ccpa' in args.features:
            assert 'shp' in args.features and 'cn' in args.features, 'ERROR!'
            ccpa = CCPA(shortest_path, path2, graph.number_of_nodes())
            ccpa1 = CCPA(shortest_path_1, path2_1, graph1.number_of_nodes())
            ccpa2 = CCPA(shortest_path_2, path2_2, graph2.number_of_nodes())
            features.append(ccpa)
            features.append(ccpa1)
            features.append(ccpa2)
            features.append(abs(ccpa1 - ccpa))
            features.append(abs(ccpa1 - ccpa2))
            features.append(abs(ccpa - ccpa2))

        if 'pr' in args.features:
            pr = graph_pr[i] + graph_pr[j]
            pr1 = graph1_pr[i] + graph1_pr[j]
            pr2 = graph2_pr[i] + graph2_pr[j]
            features.append(pr)
            features.append(pr1)
            features.append(pr2)
            features.append(abs(pr - pr1))
            features.append(abs(pr2 - pr1))
            features.append(abs(pr2 - pr))

        if 'and' in args.features:
            sum_avg_nei_deg = graph_and[i] + graph_and[j]
            sum_avg_nei_deg_1 = graph1_and[i] + graph1_and[j]
            sum_avg_nei_deg_2 = graph2_and[i] + graph2_and[j]
            features.append(sum_avg_nei_deg)
            features.append(sum_avg_nei_deg_1)
            features.append(sum_avg_nei_deg_2)
            features.append(abs(sum_avg_nei_deg - sum_avg_nei_deg_1))
            features.append(abs(sum_avg_nei_deg - sum_avg_nei_deg_2))
            features.append(abs(sum_avg_nei_deg_1 - sum_avg_nei_deg_2))

        if '_sadegh' in args.features:
            if 'and' not in args.features:
                exit(2)
            si = deg[i] * graph_and[i]
            sj = deg[j] * graph_and[j]
            si1 = deg1[i] * graph1_and[i]
            sj1 = deg1[j] * graph1_and[j]
            si2 = deg2[i] * graph2_and[i]
            sj2 = deg2[j] * graph2_and[j]
            s_e = si + sj
            s_e1 = si1 + sj1
            s_e2 = si2 + sj2
            features.append(si)
            features.append(sj)
            features.append(si1)
            features.append(sj1)
            features.append(si2)
            features.append(sj2)
            features.append(s_e)
            features.append(s_e1)
            features.append(s_e2)
            features.append(abs(s_e - s_e1))
            features.append(abs(s_e2 - s_e1))
            features.append(abs(s_e2 - s_e))


        all_features.append(np.array(features))

    if use_case == 'submit':
        with open(features_file_name, 'wb') as handle:
            pickle.dump(all_features, handle, protocol=4)

    return all_features
