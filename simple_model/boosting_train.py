import pickle
import numpy as np
import json
import os
import wandb
from utils import graph_year, args_parser, create_features

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

    wandb.init(project="unsupervised_features_newdata", entity='science4cast', mode='disabled')
    wandb.config.update(args)

    full_dynamic_graph_sparse, unconnected_vertex_pairs, pos_edge, year_start, years_delta, deg_cutoff, minedge = pickle.load(
        open(args.dataset, "rb"))

    graph_2020 = graph_year(full_dynamic_graph_sparse, 2020)
    graph_2020_delta = graph_year(full_dynamic_graph_sparse, 2020-years_delta)
    graph_2020_delta_1 = graph_year(full_dynamic_graph_sparse, 2020-years_delta-1)
    graph_2020_delta_2 = graph_year(full_dynamic_graph_sparse, 2020-years_delta-2)
    graph_2020_2delta = graph_year(full_dynamic_graph_sparse, 2020-2*years_delta)
    graph_2020_2delta_1 = graph_year(full_dynamic_graph_sparse, 2020-2*years_delta-1)
    graph_2020_2delta_2 = graph_year(full_dynamic_graph_sparse, 2020-2*years_delta-2)


    def positive_edge_extractor(pandas_row):
        return graph_2020_delta.has_edge(pandas_row[0], pandas_row[1])


    # Sample from edges!
    src = np.random.randint(0, NUM_OF_VERTICES,
                            int(len(graph_2020_delta.edges) * args.negRatio))
    dest = np.random.randint(0, NUM_OF_VERTICES,
                             int(len(graph_2020_delta.edges) * args.negRatio))
    random_edge_samples = np.array([src, dest]).T

    new_edge_2020_delta_pd = pd.DataFrame(graph_2020_delta.edges, columns=['srt', 'dest'])

    random_edge_samples_pd = pd.DataFrame(random_edge_samples,
                                          columns=['srt', 'dest'])
    random_neg_edge_samples_pd = random_edge_samples_pd[
        random_edge_samples_pd.apply(positive_edge_extractor,
                                     axis=1) == False]

    dataset = np.array(
        pd.concat([new_edge_2020_delta_pd,
                   random_neg_edge_samples_pd]).drop_duplicates())

    dataset_idx = np.unique(np.random.randint(0, len(dataset), args.samples))
    dataset_samples = dataset[dataset_idx]

    dataset_samples_sym_pd = pd.DataFrame(dataset_samples,
                                          columns=['srt', 'dest'])
    targets = np.array(dataset_samples_sym_pd.apply(positive_edge_extractor,axis=1).astype(int))
    X_train, X_test, _, _ = train_test_split(dataset_samples, targets)
    X_train = np.unique(np.concatenate([X_train, X_train[:, [1, 0]]], axis=0), axis=0)

    X_test = np.unique(np.concatenate([X_test, X_test[:, [1, 0]]], axis=0), axis=0)

    y_train = np.array(pd.DataFrame(X_train, columns=['srt', 'dest']).apply(positive_edge_extractor, axis=1).astype(int))
    y_test = np.array(pd.DataFrame(X_test, columns=['srt', 'dest']).apply(positive_edge_extractor, axis=1).astype(int))

    x_train = np.array(create_features(args, graph_2020_2delta_2, graph_2020_2delta_1, graph_2020_2delta, X_train))
    x_test = np.array(create_features(args, graph_2020_2delta_2, graph_2020_2delta_1, graph_2020_2delta, X_test))

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

    # Submit File
    print('Submission file is generating...')
    submit_features = create_features(args,
                                      graph_2020_delta_2,
                                      graph_2020_delta_1,
                                      graph_2020_delta,
                                      unconnected_vertex_pairs,
                                      use_case='submit',
                                      name=args.dataset)

    submit_pred = model.predict_proba(submit_features)

    sorted_predictions_eval = np.flip(np.argsort(submit_pred[:, 1]))

    directory = 'submit_files/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    np.save('submit_files/{}_{}_prob'.format(str(wandb.run.name), args.dataset), submit_pred)

    submit_file = directory + "{}.json".format(str(wandb.run.name))

    all_idx_list_float = list(map(float, sorted_predictions_eval))
    with open(submit_file, "w", encoding="utf8") as json_file:
        json.dump(all_idx_list_float, json_file)

    print("Solution stored as " + submit_file +
          ".\nLooking forward to your submission.")

