#taken from https://raw.githubusercontent.com/MarioKrenn6240/FutureOfAIviaAI/main/evaluate_model.py?token=GHSAT0AAAAAABHOPDRRWDIS57TS727HF56QYTIYCCQ

import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import json


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
    ypos = np.array(ypos) / max(ypos)
    xpos = np.array(xpos) / max(xpos)

    plt.plot(xpos, ypos)
    plt.show()

    AUC = sum(ROC_vals) / len(ROC_vals)
    return AUC


if __name__ == '__main__':

    delta_list = [1, 3, 5]
    cutoff_list = [25, 5, 0]
    min_edges_list = [1, 3]

    for current_delta in delta_list:
        for curr_vertex_degree_cutoff in cutoff_list:
            for current_min_edges in min_edges_list:

                data_source = "SemanticGraph_delta_" + str(current_delta) + "_cutoff_" + str(
                    curr_vertex_degree_cutoff) + "_minedge_" + str(current_min_edges)
                dataset_path = "./datasets/" + data_source + '.pkl'

                if os.path.isfile(dataset_path):
                    with open(dataset_path, "rb") as pkl_file:
                        full_dynamic_graph_sparse, unconnected_vertex_pairs, unconnected_vertex_pairs_solution, year_start, years_delta, vertex_degree_cutoff, min_edges = pickle.load(
                            pkl_file)

                    with open("logs_" + data_source + ".txt", "a") as myfile:
                        myfile.write('Read' + str(data_source) + '\n')


                    if not os.path.isfile(dataset_path):
                        continue

                    sumbit_file_path = "simple_model/submit_files/"+data_source+'.json'
                    with open(sumbit_file_path, "r") as f:
                        all_idx = json.loads(f.read())
                        all_idx = [int(x) for x in all_idx]

                    AUC = calculate_ROC(all_idx, np.array(unconnected_vertex_pairs_solution))
                    print('Area Under Curve for Evaluation: ', AUC, '\n\n\n')

                    with open("logs" + data_source[0:-4] + ".txt", "a") as log_file:
                        log_file.write("---\n")
                        log_file.write("AUC=" + str(AUC) + "\n\n")
                else:
                    print('File ', data_source, ' does not exist. Proceed to next parameter setting.')