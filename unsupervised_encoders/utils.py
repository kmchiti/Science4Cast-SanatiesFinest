import torch
import time
import numpy as np
from model import *
import networkx as nx
import random
from scipy import sparse
from datetime import date
import pickle
import argparse
import math
import json
import wandb
import os
from tqdm import tqdm


def args_parser():

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

    parser.add_argument(
        '--dataset_path',
        type=str,
        default='Science4Cast_data/CompetitionSet2017_3.pkl',
        help='Path to .pkl data (default: Science4Cast_data/CompetitionSet2017_3.pkl)'
    )
    parser.add_argument('--num_vertices',
                        type=int,
                        default=64719,
                        help='Number of vertices (default: 64719)')

    parser.add_argument('--features',
                        type=str,
                        default='baseline',
                        choices=['baseline',
                                 'aa_ja_cn_pa_ra_pr', 'ja_cn_pa_ra_pr', 'aa_cn_pa_ra_pr', 'aa_ja_pa_ra_pr', 'aa_ja_cn_ra_pr', 'aa_ja_cn_pa_pr', 'aa_ja_cn_pa_ra', 'cn_pa_ra_pr', 'ja_pa_ra_pr', 'ja_cn_ra_pr', 'ja_cn_pa_pr', 'ja_cn_pa_ra', 'aa_pa_ra_pr', 'aa_cn_ra_pr', 'aa_cn_pa_pr', 'aa_cn_pa_ra', 'aa_ja_ra_pr', 'aa_ja_pa_pr', 'aa_ja_pa_ra', 'aa_ja_cn_pr', 'aa_ja_cn_ra', 'aa_ja_cn_pa', 'pa_ra_pr', 'cn_ra_pr', 'cn_pa_pr', 'cn_pa_ra', 'ja_ra_pr', 'ja_pa_pr', 'ja_pa_ra', 'ja_cn_pr', 'ja_cn_ra', 'ja_cn_pa', 'aa_ra_pr', 'aa_pa_pr', 'aa_pa_ra', 'aa_cn_pr', 'aa_cn_ra', 'aa_cn_pa', 'aa_ja_pr', 'aa_ja_ra', 'aa_ja_pa', 'aa_ja_cn', 'ra_pr', 'pa_pr', 'pa_ra', 'cn_pr', 'cn_ra', 'cn_pa', 'ja_pr', 'ja_ra', 'ja_pa', 'ja_cn', 'aa_pr', 'aa_ra', 'aa_pa', 'aa_cn', 'aa_ja', 'pr', 'ra', 'pa', 'cn', 'ja', 'aa'],
                        help='''Features: 

                            aa: Adamic_Adar,
                            ja: JAcard,
                            cn: Common Neighbours,
                            pa: Preferential Attachment,
                            ra: Resource Allocation,
                            pr: Page Rank.
                            
                        All schemes should be seperated by '_'.
                        (Example: 'aa_ja' will add adamic_adar and jacard.)
                        ''')

    parser.add_argument('--train_frac',
                        type=float,
                        default=0.9,
                        help='Fraction of data for train (default: 0.9)')

    parser.add_argument('--seed',
                        type=int,
                        default=12,
                        help='seed (default: 12)')

    # Optimizer

    parser.add_argument('--lr',
                        type=float,
                        default=5 * 10**-4,
                        help='Initial LR (default : 5*10**-4)')

    parser.add_argument('--max_iter',
                        type=int,
                        default=10000,
                        help='Maximum Iterations (default : 10000)')

    parser.add_argument('--batch-size',
                        type=int,
                        default=100,
                        help='Train batch size (default: 100)')

    parser.add_argument('--test-batch-size',
                        type=int,
                        default=100,
                        help='Test batch size (default: 100)')

    args = parser.parse_args()

    args.test_frac = 1 - args.train_frac

    return args


def create_training_data(full_graph,
                         year_start,
                         years_delta,
                         edges_used=500000,
                         vertex_degree_cutoff=10):
    """
    Create a graph from the data for a specific year interval

    :param full_graph: Full graph, numpy array dim(n,3) [vertex 1, vertex 2, time stamp]
    :param year_start: year of graph
    :param years_delta: distance for prediction in years (prediction on graph of year_start+years_delta)
    :param edges_used: optional filter to create a random subset of edges for rapid prototyping (default: 500,000)
    :param vertex_degree_cutoff: optional filter, for vertices in training set having a minimal degree of at least vertex_degree_cutoff  (default: 10)
    :return:

    all_edge_list: graph of year_start, numpy array dim(n,2)
    unconnected_vertex_pairs: potential edges for year_start+years_delta
    unconnected_vertex_pairs_solution: numpy array with integers (0=unconnected, 1=connected), solution, length = len(unconnected_vertex_pairs)
    """

    NUM_OF_VERTICES = 64719  # number of vertices of the semantic net
    years = [year_start, year_start + years_delta]
    day_origin = date(1990, 1, 1)

    all_G = []
    all_edge_lists = []
    all_sparse = []
    for yy in years:
        print('    Create Graph for ', yy)
        day_curr = date(yy, 12, 31)
        all_edges_curr = full_graph[full_graph[:, 2] < (day_curr -
                                                        day_origin).days]
        adj_mat_sparse_curr = sparse.csr_matrix(
            (np.ones(len(all_edges_curr)),
             (all_edges_curr[:, 0], all_edges_curr[:, 1])),
            shape=(NUM_OF_VERTICES, NUM_OF_VERTICES))
        G_curr = nx.from_scipy_sparse_matrix(adj_mat_sparse_curr,
                                             parallel_edges=False,
                                             create_using=None,
                                             edge_attribute='weight')

        all_G.append(G_curr)
        all_sparse.append(adj_mat_sparse_curr)
        all_edge_lists.append(all_edges_curr)

        print('    Done: Create Graph for ', yy)
        print('    num of edges: ', G_curr.number_of_edges())

    all_degs = np.array(all_sparse[0].sum(0))[0]

    ## Create all edges to be predicted
    all_vertices = np.array(range(NUM_OF_VERTICES))
    vertex_large_degs = all_vertices[
        all_degs >=
        vertex_degree_cutoff]  # use only vertices with degrees larger than 10.

    unconnected_vertex_pairs = []
    unconnected_vertex_pairs_solution = []

    time_start = time.time()
    while len(unconnected_vertex_pairs) < edges_used:
        v1, v2 = random.sample(range(len(vertex_large_degs)), 2)

        if v1 != v2 and not all_G[0].has_edge(v1, v2):
            if len(unconnected_vertex_pairs) % 10**6 == 0:
                time_end = time.time()
                print('    edge progress (', time_end - time_start, 'sec): ',
                      len(unconnected_vertex_pairs) / 10**6, 'M/',
                      edges_used / 10**6, 'M')
                time_start = time.time()
            unconnected_vertex_pairs.append((v1, v2))
            unconnected_vertex_pairs_solution.append(all_G[1].has_edge(v1, v2))

    print('Number of unconnected vertex pairs for prediction: ',
          len(unconnected_vertex_pairs_solution))
    print('Number of vertex pairs that will be connected: ',
          sum(unconnected_vertex_pairs_solution))
    print(
        'Ratio of vertex pairs that will be connected: ',
        sum(unconnected_vertex_pairs_solution) /
        len(unconnected_vertex_pairs_solution))

    unconnected_vertex_pairs = np.array(unconnected_vertex_pairs)
    unconnected_vertex_pairs_solution = np.array(
        list(map(int, unconnected_vertex_pairs_solution)))
    all_edge_list = np.array(all_edge_lists[0])

    return all_edge_list, unconnected_vertex_pairs, unconnected_vertex_pairs_solution


def compute_all_properties(all_sparse,
                           graph_0, graph_1, graph_2,
                           deg_0, deg_1, deg_2,
                           AA02, AA12, AA22,
                           all_degs0, all_degs1, all_degs2,
                           all_degs02, all_degs12, all_degs22,
                            graph0_pr, graph1_pr, graph2_pr,
                           v1,
                           v2,
                           args,
                           ):
    """
    Computes hand-crafted properties for one vertex in vlist
    """
    all_properties = []

    all_properties.append(all_degs0[v1])  # 0
    all_properties.append(all_degs0[v2])  # 1
    all_properties.append(all_degs1[v1])  # 2
    all_properties.append(all_degs1[v2])  # 3
    all_properties.append(all_degs2[v1])  # 4
    all_properties.append(all_degs2[v2])  # 5
    all_properties.append(all_degs02[v1])  # 6
    all_properties.append(all_degs02[v2])  # 7
    all_properties.append(all_degs12[v1])  # 8
    all_properties.append(all_degs12[v2])  # 9
    all_properties.append(all_degs22[v1])  # 10
    all_properties.append(all_degs22[v2])  # 11

    all_properties.append(AA02[v1, v2])  # 12
    all_properties.append(AA12[v1, v2])  # 13
    all_properties.append(AA22[v1, v2])  # 14

    all_properties.append(deg_0[v1] + deg_0[v2])
    all_properties.append(deg_1[v1] + deg_1[v2])
    all_properties.append(deg_2[v1] + deg_2[v2])

    # # Networkx graph obj
    # graph_1 = nx.from_scipy_sparse_matrix(all_sparse[0])
    # graph_2 = nx.from_scipy_sparse_matrix(all_sparse[1])
    # graph_3 = nx.from_scipy_sparse_matrix(all_sparse[2])

    # adamic_adar
    if 'aa' in args.features:
        all_properties.append(list(nx.adamic_adar_index(graph_0, [(v1,v2)]))[0][2]) #15
        all_properties.append(list(nx.adamic_adar_index(graph_1, [(v1, v2)]))[0][2]) #16
        all_properties.append(list(nx.adamic_adar_index(graph_2, [(v1, v2)]))[0][2]) #17

    # jacard
    if 'ja' in args.features:
        all_properties.append(list(nx.jaccard_coefficient(graph_0, [(v1, v2)]))[0][2])  # 18
        all_properties.append(list(nx.jaccard_coefficient(graph_1, [(v1, v2)]))[0][2])  # 19
        all_properties.append(list(nx.jaccard_coefficient(graph_2, [(v1, v2)]))[0][2])  # 20

    # Common neighbours
    if 'cn' in args.features:
        all_properties.append(len(list(nx.common_neighbors(graph_0, v1, v2)))) #18
        all_properties.append(len(list(nx.common_neighbors(graph_1, v1, v2)))) #18
        all_properties.append(len(list(nx.common_neighbors(graph_2, v1, v2)))) #18

    if 'pa' in args.features:
        all_properties.append( len(list(nx.neighbors(graph_0, v1))) * len(list(nx.neighbors(graph_0, v2))))
        all_properties.append( len(list(nx.neighbors(graph_1, v1))) * len(list(nx.neighbors(graph_1, v2))))
        all_properties.append( len(list(nx.neighbors(graph_2, v1))) * len(list(nx.neighbors(graph_2, v2))))

    if 'ra' in args.features:

        ra_0 = 0
        for z in list(nx.common_neighbors(graph_0, v1, v2)):
            ra_0 += 1/len(list(nx.neighbors(graph_0, z)))

        ra_1 = 0
        for z in list(nx.common_neighbors(graph_1, v1, v2)):
            ra_1 += 1/len(list(nx.neighbors(graph_1, z)))

        ra_2 = 0
        for z in list(nx.common_neighbors(graph_2, v1, v2)):
            ra_2 += 1/len(list(nx.neighbors(graph_2, z)))

        all_properties.append(ra_0)
        all_properties.append(ra_1)
        all_properties.append(ra_2)

    if 'pr' in args.features:
        all_properties.append(graph0_pr[v1]+graph0_pr[v2])
        all_properties.append(graph1_pr[v1]+graph1_pr[v2])
        all_properties.append(graph2_pr[v1]+graph2_pr[v2])

    # all_properties.append(katz_centerality_1[v1])  #18
    # all_properties.append(katz_centerality_2[v1])  #19
    # all_properties.append(katz_centerality_3[v1])  #20
    # all_properties.append(katz_centerality_1[v2])  #21
    # all_properties.append(katz_centerality_2[v2])  #22
    # all_properties.append(katz_centerality_3[v2])  #23
    #
    # all_properties.append(sim_rank_1[v1][v2]) #24
    # all_properties.append(sim_rank_2[v1][v2]) #25
    # all_properties.append(sim_rank_3[v1][v2]) #26

    return all_properties


def compute_all_properties_of_list(all_sparse, vlist, args):
    """
    Computes hand-crafted properties for all vertices in vlist
    """
    time_start = time.time()
    AA02 = all_sparse[0]
    AA02 = AA02 / AA02.max()
    AA12 = all_sparse[1]
    AA12 = AA12 / AA12.max()
    AA22 = all_sparse[2]
    AA22 = AA02 / AA22.max()

    all_degs0 = np.array(all_sparse[0].sum(0))[0]
    if np.max(all_degs0) > 0:
        all_degs0 = all_degs0 / np.max(all_degs0)

    all_degs1 = np.array(all_sparse[1].sum(0))[0]
    if np.max(all_degs1) > 0:
        all_degs1 = all_degs1 / np.max(all_degs1)

    all_degs2 = np.array(all_sparse[2].sum(0))[0]
    if np.max(all_degs2) > 0:
        all_degs2 = all_degs2 / np.max(all_degs2)

    all_degs02 = np.array(AA02[0].sum(0))[0]
    if np.max(all_degs2) > 0:
        all_degs02 = all_degs02 / np.max(all_degs02)

    all_degs12 = np.array(AA12[1].sum(0))[0]
    if np.max(all_degs12) > 0:
        all_degs12 = all_degs12 / np.max(all_degs12)

    all_degs22 = np.array(AA22[2].sum(0))[0]
    if np.max(all_degs22) > 0:
        all_degs22 = all_degs22 / np.max(all_degs22)

    # Networkx graph obj
    print('create graph')
    graph_0 = nx.from_scipy_sparse_matrix(all_sparse[0])
    graph_1 = nx.from_scipy_sparse_matrix(all_sparse[1])
    graph_2 = nx.from_scipy_sparse_matrix(all_sparse[2])

    # phi = (1 + math.sqrt(5)) / 2.0  # TODO add to parsers
    # katz_centerality_1 = nx.katz_centrality(graph_1, 1 / phi - 0.01)
    # katz_centerality_2 = nx.katz_centrality(graph_2, 1 / phi - 0.01)
    # katz_centerality_3 = nx.katz_centrality(graph_3, 1 / phi - 0.01)

    # sim_rank_1 = nx.simrank_similarity(graph_1)
    # sim_rank_2 = nx.simrank_similarity(graph_2)
    # sim_rank_3 = nx.simrank_similarity(graph_3)
    # print('finish')
    all_properties = []

    deg_0 = graph_0.degree()
    deg_1 = graph_1.degree()
    deg_2 = graph_2.degree()

    if 'pr' in args.features:
        graph0_pr = nx.pagerank(graph_0)
        graph1_pr = nx.pagerank(graph_1)
        graph2_pr = nx.pagerank(graph_2)
    else:
        graph0_pr = {}
        graph1_pr = {}
        graph2_pr = {}

    print('    Computed all matrix squares, ready to ruuuumbleeee...')

    for ii in tqdm(range(len(vlist))):
        vals = compute_all_properties(all_sparse,
                                      graph_0, graph_1, graph_2,
                                      deg_0, deg_1, deg_2,
                                      AA02, AA12, AA22, all_degs0,
                                      all_degs1, all_degs2, all_degs02,
                                      all_degs12, all_degs22,
                                      graph0_pr, graph1_pr, graph2_pr,
                                      vlist[ii][0],
                                      vlist[ii][1],
                                      args)

        all_properties.append(vals)
        if ii % 10**5 == 0:
            print('compute features: (',
                  time.time() - time_start, 'sec) ', ii / 10**6, 'M/',
                  len(vlist) / 10**6, 'M')
            time_start = time.time()
    print("finish")
    print(all_properties[0])
    return all_properties


def data_generator(args):
    '''
    Generate train and test data to train
    :param args
    :return: data_train0, data_train1, data_test0, data_test1 (positive/negative data for train/test)
    '''

    full_dynamic_graph_sparse, unconnected_vertex_pairs, year_start, years_delta = pickle.load(
        open(args.dataset_path, "rb"))

    print(args.dataset_path + ' has ' + str(len(full_dynamic_graph_sparse)) +
          ' edges between a total of ' + str(args.num_vertices) +
          ' vertices.\n\n')
    print(
        'The goal is to predict which of ' +
        str(len(unconnected_vertex_pairs)) +
        ' unconnectedvertex-pairs\nin unconnected_vertex_pairs will be connected until '
        + str(year_start + years_delta) + '.')

    edges_used = 1 * 10**6  # Best would be to use all vertices, to create more training data. But that takes long and requires huge amount of memory. So here we use a random subset.
    vertex_degree_cutoff = 10
    train_dynamic_graph_sparse, train_edges_for_checking, train_edges_solution = create_training_data(
        full_dynamic_graph_sparse,
        year_start - years_delta,
        years_delta,
        edges_used=edges_used,
        vertex_degree_cutoff=vertex_degree_cutoff)

    day_origin = date(1990, 1, 1)
    years = [year_start - 3, year_start - 4, year_start - 5]

    train_sparse = []
    for yy in years:
        print('    Create Graph for ', yy)
        day_curr = date(yy, 12, 31)
        train_edges_curr = train_dynamic_graph_sparse[
            train_dynamic_graph_sparse[:, 2] < (day_curr - day_origin).days]
        adj_mat_sparse_curr = sparse.csr_matrix(
            (np.ones(len(train_edges_curr)),
             (train_edges_curr[:, 0], train_edges_curr[:, 1])),
            shape=(args.num_vertices, args.num_vertices))

        train_sparse.append(adj_mat_sparse_curr)

    print('    Shuffle training data...')
    train_valid_test_size = [args.train_frac, args.test_frac, 0.0]
    x = [i
         for i in range(len(train_edges_for_checking))]  # random shuffle input

    random.shuffle(x)
    train_edges_for_checking = train_edges_for_checking[x]
    train_edges_solution = train_edges_solution[x]

    print('    Split dataset...')
    idx_traintest = int(
        len(train_edges_for_checking) * train_valid_test_size[0])

    data_edges_train = train_edges_for_checking[0:idx_traintest]
    solution_train = train_edges_solution[0:idx_traintest]

    data_edges_test = train_edges_for_checking[idx_traintest:]
    solution_test = train_edges_solution[idx_traintest:]

    print('Training, connected  : ', sum(solution_train == 1))
    print('Training, unconnected: ', sum(solution_train == 0))

    # Rather than using all connected and unconnected vertex pairs for training
    # (i.e. needing to compute their properties), we reject about 99% of all unconnected
    # examples, to have more examples of connected cases in the training. This significantly
    # speeds up the computation, at the price of precision.
    data_edges_train_smaller = []
    solution_train_smaller = []
    for ii in range(len(data_edges_train)):
        if (solution_train[ii] == 0
                and random.random() < 0.01) or solution_train[ii] == 1:
            data_edges_train_smaller.append(data_edges_train[ii])
            solution_train_smaller.append(solution_train[ii])

    data_train = compute_all_properties_of_list(train_sparse,
                                                data_edges_train_smaller,
                                                args)

    data_train0 = []
    data_train1 = []
    for ii in range(len(data_edges_train_smaller)):
        if solution_train_smaller[ii] == 1:
            data_train1.append(data_train[ii])
        else:
            data_train0.append(data_train[ii])

    data_test = compute_all_properties_of_list(train_sparse,
                                               data_edges_test,
                                               args)
    data_test0 = []
    data_test1 = []
    for ii in range(len(data_edges_test)):
        if solution_test[ii] == 1:
            data_test1.append(data_test[ii])
        else:
            data_test0.append(data_test[ii])

    # TODO: generate batch here

    device = torch.device("cpu")
    data_train0 = torch.tensor(data_train0, dtype=torch.float).to(device)
    data_test0 = torch.tensor(data_test0, dtype=torch.float).to(device)

    data_train1 = torch.tensor(data_train1, dtype=torch.float).to(device)
    data_test1 = torch.tensor(data_test1, dtype=torch.float).to(device)

    return data_train0, data_train1, data_test0, data_test1, solution_train_smaller, solution_test, data_train, data_test


import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def flatten(t):
    return [item for sublist in t for item in sublist]


def calculate_ROC(data_vertex_pairs, data_solution):
    data_solution = np.array(data_solution)
    data_vertex_pairs_sorted = data_solution[data_vertex_pairs]

    xpos = [0]
    ypos = [0]
    ROC_vals = []
    for ii in range(len(data_vertex_pairs_sorted)):
        if data_vertex_pairs_sorted[ii] == 1:
            xpos.append(xpos[-1])
            ypos.append(ypos[-1] + 1)
        if data_vertex_pairs_sorted[ii] == 0:
            xpos.append(xpos[-1] + 1)
            ypos.append(ypos[-1])
            ROC_vals.append(ypos[-1])

        # # # # # # # # # # # # # # #
        #
        # We normalize the ROC curve such that it starts at (0,0) and ends at (1,1).
        # Then our final metric of interest is the Area under that curve.
        # AUC is between [0,1].
        # AUC = 0.5 is acchieved by random predictions
        # AUC = 1.0 stands for perfect prediction.

    ROC_vals = np.array(ROC_vals) / max(ypos)

    AUC = sum(ROC_vals) / len(ROC_vals)
    return AUC


def create_submit(args, model):
    day_origin = date(1990, 1, 1)
    device = torch.device("cpu")
    model = model.to(device)

    directory = 'submit_files/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    submit_file =  directory  + str(wandb.run.name)  + ".json"


    # data_source = 'Science4Cast_data/TrainSet2014_3.pkl'
    data_source = 'Science4Cast_data/CompetitionSet2017_3.pkl'
    full_dynamic_graph_sparse, unconnected_vertex_pairs, year_start, years_delta = pickle.load(open(data_source, "rb"))
    year_start = 2017
    print('2) Makes predictions for ' + str(year_start) + ' -> ' + str(year_start + 3) + ' data.')
    years = [year_start, year_start - 1, year_start - 2]

    print('2.1) Computes the 15 properties for the ' + str(year_start) + ' data.')
    eval_sparse = []
    for yy in years:
        print('    Create Graph for ', yy)
        day_curr = date(yy, 12, 31)
        eval_edges_curr = full_dynamic_graph_sparse[full_dynamic_graph_sparse[:, 2] < (day_curr - day_origin).days]
        adj_mat_sparse_curr = sparse.csr_matrix(
            (np.ones(len(eval_edges_curr)), (eval_edges_curr[:, 0], eval_edges_curr[:, 1])),
            shape=(args.num_vertices, args.num_vertices)
        )

        eval_sparse.append(adj_mat_sparse_curr)

    print('compute all properties for evaluation')
    eval_examples = compute_all_properties_of_list(eval_sparse, unconnected_vertex_pairs, args)
    eval_examples = np.array(eval_examples)

    print('2.2) Uses the trained network to predict whether edges are created by ' + str(year_start + 3) + '.')
    eval_examples = torch.tensor(eval_examples, dtype=torch.float).to(device)
    all_predictions_eval = flatten(model(eval_examples).detach().cpu().numpy())

    print(
        '3) Creates a sorted index list, from highest predicted vertex pair to least predicted one (sorted_predictions)')
    sorted_predictions_eval = np.flip(np.argsort(all_predictions_eval, axis=0))


    
    all_idx_list_float = list(map(float, sorted_predictions_eval))
    with open(submit_file, "w", encoding="utf8") as json_file:
        json.dump(all_idx_list_float, json_file)

    print("Solution stored as " + submit_file + ".\nLooking forward to your submission.")
