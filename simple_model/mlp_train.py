
import pickle
import numpy as np
import torch.nn
from scipy import sparse
import networkx as nx
from datetime import date
from scipy import sparse
from datetime import date
import json
import os
from tqdm import tqdm
import wandb
from utils import graph_year, args_parser, create_features
from mlp import  *

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import torch.utils.data as data_utils
from torch.utils.data import DataLoader


NUM_OF_VERTICES=64719

if __name__ == '__main__':

    torch.manual_seed(12) #In the memory of DAVAZA MASUM!

    args = args_parser()

    if 'ccpa' in args.features:
        assert 'cn' in args.features and 'shp' in args.features, 'ERROR!'

    wandb.init(project="unsipervised_features", entity='science4cast')
    wandb.config.update(args)

    wandb.run.name += '_mlp'

    full_dynamic_graph_sparse, unconnected_vertex_pairs, year_start, years_delta = pickle.load(open(args.dataset, "rb"))

    graph_2014 = graph_year(full_dynamic_graph_sparse, 2014)
    graph_2015 = graph_year(full_dynamic_graph_sparse, 2015)
    graph_2016 = graph_year(full_dynamic_graph_sparse, 2016)
    graph_2017 = graph_year(full_dynamic_graph_sparse, 2017)

    new_edge_2017 = [i for i in graph_2017.edges if i not in graph_2016.edges]

    # Sample from edges!
    src = np.random.randint(0, NUM_OF_VERTICES,  len(new_edge_2017))
    trg = np.random.randint(0, NUM_OF_VERTICES,  len(new_edge_2017))
    edge_samples = np.array([src,trg]).T
    edge_samples = np.unique(np.concatenate([new_edge_2017, edge_samples]), axis=0)

    dataset_idx = np.unique(np.random.randint(0, len(edge_samples), args.samples))
    edge_samples = edge_samples[dataset_idx]

    target = []
    for i, j in tqdm(edge_samples, desc="Creating the Target Set"):
        if graph_2017.has_edge(i, j):
            target.append(1)
        else:
            target.append(0)

    X_train, X_test, y_train, y_test = train_test_split(edge_samples, target)

    x_train = np.array(create_features(args, graph_2014, graph_2016, X_train))
    x_test = np.array(create_features(args, graph_2014, graph_2016, X_test))
    x_test = torch.tensor(x_test, dtype=torch.float)
    y_test = torch.tensor(y_test, dtype=torch.float)
    x_train = torch.tensor(x_train, dtype=torch.float)
    y_train = torch.tensor(y_train, dtype=torch.float)

    train = data_utils.TensorDataset(x_train, y_train)
    train_loader = data_utils.DataLoader(train, batch_size=args.bs, shuffle=True)

    test = data_utils.TensorDataset(x_test, y_test)
    test_loader = data_utils.DataLoader(test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = torch.nn.MSELoss()
    model = ff_network(features_num=x_test.shape[-1], hidden_layer=args.hiddenLayers).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr/2000)


    for epoch in tqdm(range(args.epoch), desc="Training"):
        epoch_loss = 0.0
        for features, targets in train_loader:
            features.to(device)
            features.to(targets)

            output = model(features)
            loss = criterion(output, targets)
            epoch_loss += loss
            loss.backward()
            optimizer.step()
        print('epoch_loss:', epoch_loss)
        wandb.log({'Train Loss': epoch_loss,
                   "Epoch": epoch})

    wandb.log({
               "AUC_test": roc_auc_score(y_test.detach().numpy(),
                                         model(x_test).detach().numpy())})


    # Submit File
    print('Submission Features are generating...')
    submit_features = create_features(args,
                               graph_2015, graph_2017,
                               unconnected_vertex_pairs,
                               use_case='submit')
    submit_features_tensor = torch.tensor(submit_features, dtype=torch.float)

    submit_pred = model(submit_features_tensor).detach().numpy()
    sorted_predictions_eval = np.flip(np.argsort(submit_pred))


    print('Submission file is generating...')
    directory = 'submit_files/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    submit_file =  directory  + "{}.json".format(str(wandb.run.name))

    all_idx_list_float = list(map(float, sorted_predictions_eval))
    with open(submit_file, "w", encoding="utf8") as json_file:
        json.dump(all_idx_list_float, json_file)

    print("Solution stored as " + submit_file + ".\nLooking forward to your submission.")

