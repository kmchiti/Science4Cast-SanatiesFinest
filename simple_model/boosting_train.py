import pickle
import numpy as np
from scipy import sparse
import networkx as nx
from datetime import date
from scipy import sparse
from datetime import date
import json
import os
from tqdm import tqdm
import wandb
from utils import graph_year, args_parser, create_features, graph_year2

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import pandas as pd

NUM_OF_VERTICES = 64719

if __name__ == '__main__':

    args = args_parser()

    if 'ccpa' in args.features:
        if 'cn' not in args.features:
            args.features += '_cn'
        if 'shp' not in args.features:
            args.features += '_shp'

    wandb.init(project="unsipervised_features", entity='science4cast', mode='disabled')
    wandb.config.update(args)

    wandb.run.name += '_4graphs'

    full_dynamic_graph_sparse, unconnected_vertex_pairs, pos_edge, year_start, years_delta, deg_cutoff, minedge = pickle.load(
        open(args.dataset, "rb"))

    graph_2014 = graph_year(full_dynamic_graph_sparse, 2014)
    graph_2015 = graph_year(full_dynamic_graph_sparse, 2015)
    graph_2016 = graph_year(full_dynamic_graph_sparse, 2016)
    graph_2017_half = graph_year2(full_dynamic_graph_sparse, 2017,
                                  args.borj2017)
    graph_2017 = graph_year(full_dynamic_graph_sparse, 2017)

    def positive_edge_extractor_2017(pandas_row):
        return graph_2017.has_edge(pandas_row[0], pandas_row[1])

    new_edge_2017 = [
        i for i in graph_2017.edges if i not in graph_2017_half.edges
    ]

    # Sample from edges!
    src = np.random.randint(0, NUM_OF_VERTICES,
                            int(len(new_edge_2017) * args.negRatio))
    dest = np.random.randint(0, NUM_OF_VERTICES,
                             int(len(new_edge_2017) * args.negRatio))
    random_edge_samples = np.array([src, dest]).T

    new_edge_2017_pd = pd.DataFrame(new_edge_2017, columns=['srt', 'dest'])

    random_edge_samples_pd = pd.DataFrame(random_edge_samples,
                                          columns=['srt', 'dest'])
    random_neg_edge_samples_pd = random_edge_samples_pd[
        random_edge_samples_pd.apply(positive_edge_extractor_2017,
                                     axis=1) == False]

    dataset = np.array(
        pd.concat([new_edge_2017_pd,
                   random_neg_edge_samples_pd]).drop_duplicates())

    dataset_idx = np.unique(np.random.randint(0, len(dataset), args.samples))
    dataset_samples = dataset[dataset_idx]
    # dataset_samples_sym = np.concatenate(
    #     [dataset_samples, dataset_samples[:, [1, 0]]], axis=0)
    dataset_samples_sym_pd = pd.DataFrame(dataset_samples,
                                          columns=['srt', 'dest'])
    targets = np.array(
        dataset_samples_sym_pd.apply(positive_edge_extractor_2017,
                                     axis=1).astype(int))
    X_train, X_test, y_train, y_test = train_test_split(
        dataset_samples, targets)
    X_train = np.unique(np.concatenate(
        [X_train, X_train[:, [1, 0]]], axis=0), axis=0)

    X_test = np.unique(np.concatenate(
    [X_test, X_test[:, [1, 0]]], axis=0), axis=0)

    y_train = np.array(
        pd.DataFrame(X_train,
                     columns=['srt', 'dest']).apply(positive_edge_extractor_2017, axis=1).astype(int))
    y_test = np.array(
        pd.DataFrame(X_test,
                     columns=['srt', 'dest']).apply(positive_edge_extractor_2017, axis=1).astype(int))

    x_train = np.array(
        create_features(args, graph_2014, graph_2015, graph_2016, graph_2017_half,
                        X_train))
    x_test = np.array(
        create_features(args, graph_2014, graph_2015, graph_2016, graph_2017_half, X_test))

    model = GradientBoostingClassifier(n_estimators=args.estimators,
                                       learning_rate=args.lr,
                                       loss=args.loss,
                                       max_depth=args.depth,
                                       subsample=args.subsamples,
                                       min_samples_leaf=args.sampleLeaf,
                                       min_samples_split=args.minSampleSplit)

    model.fit(x_train, y_train)
    auc_test = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])

    score = {"SCORE": model.score(x_test, y_test), "AUC_test": auc_test}
    wandb.log(score)
    print(score)

    if auc_test < 0.9075:
        exit(-1)

    # Submit File
    print('Submission file is generating...')
    submit_features = create_features(args,
                                      graph_2014,
                                      graph_2015,
                                      graph_2016,
                                      graph_2017,
                                      unconnected_vertex_pairs,
                                      use_case='submit')

    submit_pred = model.predict_proba(submit_features)

    sorted_predictions_eval = np.flip(np.argsort(submit_pred[:, 1]))

    directory = 'submit_files/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    np.save('submit_files/{}_prob'.format(str(wandb.run.name)), submit_pred)

    submit_file = directory + "{}.json".format(str(wandb.run.name))

    all_idx_list_float = list(map(float, sorted_predictions_eval))
    with open(submit_file, "w", encoding="utf8") as json_file:
        json.dump(all_idx_list_float, json_file)

    print("Solution stored as " + submit_file +
          ".\nLooking forward to your submission.")

