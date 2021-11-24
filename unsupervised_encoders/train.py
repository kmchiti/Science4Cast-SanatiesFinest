import torch
import time
import numpy as np
from model import *
import networkx as nx
import random
from scipy import sparse
from datetime import date
import pickle
from utils import *
import wandb
import json

def train(args, model, optimizer, data_train0, data_train1, data_test0,
          data_test1, solution_train_smaller, solution_test, data_train, data_test):

    size_of_loss_check = 2000  #todo: add this to arg_parser()
    test_loss_total = []
    moving_avg = []
    criterion = torch.nn.MSELoss()

    for iteration in range(
            args.max_iter
    ):  # should be much larger, with good early stopping criteria
        model.train()
        data_sets = [data_train0, data_train1]
        total_loss = 0
        for idx_dataset in range(len(data_sets)):
            idx = torch.randint(0, len(data_sets[idx_dataset]),
                                (args.batch_size, ))
            data_train_samples = data_sets[idx_dataset][idx]
            calc_properties = model(data_train_samples)
            curr_pred = torch.tensor([idx_dataset] * args.batch_size,
                                     dtype=torch.float).to(device)
            real_loss = criterion(calc_properties, curr_pred)
            total_loss += torch.clamp(real_loss, min=0., max=50000.).double()

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Evaluating the current quality.
        with torch.no_grad():
            model.eval()
            # calculate train set
            eval_datasets = [data_train0, data_train1, data_test0, data_test1]
            all_real_loss = []
            for idx_dataset in range(len(eval_datasets)):
                eval_datasets[idx_dataset]
                calc_properties = model(
                    eval_datasets[idx_dataset][0:size_of_loss_check])
                curr_pred = torch.tensor(
                    [idx_dataset % 2] *
                    len(eval_datasets[idx_dataset][0:size_of_loss_check]),
                    dtype=torch.float).to(device)
                real_loss = criterion(calc_properties, curr_pred)
                all_real_loss.append(real_loss.detach().cpu().numpy())

            test_loss_total.append(
                np.mean(all_real_loss[2]) + np.mean(all_real_loss[3]))

            if iteration % 50 == 0:
                wandb.log({
                    'iteration': iteration,
                    'train_loss':np.mean(all_real_loss[0]) + np.mean(all_real_loss[1]),
                    'test_loss': np.mean(all_real_loss[2]) + np.mean(all_real_loss[3])
                })
                print(
                    str(iteration) + ' - train: ',
                    np.mean(all_real_loss[0]) + np.mean(all_real_loss[1]),
                    '; test: ',
                    np.mean(all_real_loss[2]) + np.mean(all_real_loss[3]))

            if len(test_loss_total) > 10000:  # early stopping
                test_loss_moving_avg = sum(test_loss_total[-50:])
                moving_avg.append(test_loss_moving_avg)
                if len(moving_avg) > 100:
                    if moving_avg[-1] > moving_avg[-2] and moving_avg[
                            -1] > moving_avg[-10]:
                        print('Early stopping kicked in')
                        break

    data_train = torch.tensor(data_train, dtype=torch.float).to(device)
    all_predictions_train = flatten(model(data_train).detach().cpu().numpy())
    sorted_predictions_train = np.flip(np.argsort(all_predictions_train, axis=0))
    AUC_train = calculate_ROC(sorted_predictions_train, solution_train_smaller)
    print('    AUC_train: ', AUC_train)

    data_test = torch.tensor(data_test, dtype=torch.float).to(device)
    all_predictions_test = flatten(model(data_test).detach().cpu().numpy())
    sorted_predictions_test = np.flip(np.argsort(all_predictions_test, axis=0))
    AUC_test = calculate_ROC(sorted_predictions_test, solution_test)
    print('    AUC_test: ', AUC_test)

    wandb.log({"AUC_trian": AUC_train,
               "AUC_test": AUC_test})
    if AUC_test < 0.8:
        exit(2)



if __name__ == '__main__':
    args = args_parser()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    wandb.init( project="unsipervised_features", entity='science4cast')
    wandb.config.update(args)

    device = torch.device("cpu")

    data_train0, data_train1, data_test0, data_test1, solution_train_smaller, solution_test, data_train, data_test = data_generator(args)

    print(data_train0.shape, data_train1.shape, data_test0.shape,
          data_test1.shape)

    model = model_generator(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train(args, model, optimizer, data_train0, data_train1, data_test0,
          data_test1, solution_train_smaller, solution_test, data_train, data_test)

    create_submit(args, model)

