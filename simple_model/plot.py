from scipy import sparse
from datetime import date
import numpy as np
import argparse
import networkx as nx
from tqdm import tqdm
import os
import pickle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from utils import graph_year, args_parser, create_features


class myargs:
    def __init__(self):
        self.features = 'cn_pr_and_ja_nd_ccpa_shp'
        self.loss = 'deviance'
        self.depth = 5
        self.epoch = 100
        self.lr = 0.05
        self.minSampleSplit = 14
        self.samples = 450000
        self.sampleLeaf = 3
        self.subsamples = 0.5
        self.dataset = '../CompetitionSet2017_3.pkl' 
        self.borj2017 = 10  
        self.negRatio = 4
args = myargs()

full_dynamic_graph_sparse, unconnected_vertex_pairs, year_start, years_delta = pickle.load(
        open(args.dataset, "rb"))

graph_2014 = graph_year(full_dynamic_graph_sparse, 2014)
graph_2015 = graph_year(full_dynamic_graph_sparse, 2015)
graph_2016 = graph_year(full_dynamic_graph_sparse, 2016)
graph_2017 = graph_year(full_dynamic_graph_sparse, 2017)
Features = create_features(args, graph_2015, graph_2016, graph_2017, unconnected_vertex_pairs, use_case='train')
Features = np.array(Features)
np.savetxt("Features.csv", Features, delimiter=",")

ja_2015 = Features[:,24]
ja_2016 = Features[:,25]
ja_2017 = Features[:,26]
cn_2015 = Features[:,30]
cn_2016 = Features[:,31]
cn_2017 = Features[:,32]
shp_2015 = Features[:,36]
shp_2016 = Features[:,37]
shp_2017 = Features[:,38]
ccpa_2015 = Features[:,42]
ccpa_2016 = Features[:,43]
pr_2015 = Features[:,47]
pr_2016 = Features[:,48]
pr_2017 = Features[:,49]
and_2015 = Features[:,53]
and_2016 = Features[:,54]
and_2017 = Features[:,55]

def plot_Deg_Dist(graph_fisrtYear, graph_secondYear, graph_thirdYear, step, fisrtLabel = "2014", secondLabel = "2015", ThirdLabel = "2016"):
  degs_fisrtYear = np.array(graph_fisrtYear.degree())[:,1]
  degs_secondYear = np.array(graph_secondYear.degree())[:,1]
  degs_thirdYear = np.array(graph_thirdYear.degree())[:,1]
  delta_year1 = np.max(degs_fisrtYear)//step
  delta_year2 = np.max(degs_secondYear)//step
  delta_year3 = np.max(degs_thirdYear)//step
  dist_year1 = []
  dist_year2 = []
  dist_year3 = []

  for i in range(int(step*delta_year3/delta_year1)+1):
    dist_year1.append(len(degs_fisrtYear[(degs_fisrtYear<(i+1)*delta_year1) * (degs_fisrtYear>(i)*delta_year1)]))
    dist_year2.append(len(degs_secondYear[(degs_secondYear<(i+1)*delta_year1) * (degs_secondYear>(i)*delta_year1)]))
    dist_year3.append(len(degs_thirdYear[(degs_thirdYear<(i+1)*delta_year1) * (degs_thirdYear>(i)*delta_year1)]))

  sns.set()
  width = 0.35  # the width of the bars
  x_labels_st = np.arange(0,int(step*delta_year3/delta_year1)+1)*(delta_year1)
  x_labels = np.array([str(x_) for x_ in x_labels_st])
  fig = plt.figure(figsize=(step,3))
  ax = fig.add_axes([0,0,1,1])
  ax.bar(np.arange(len(x_labels))-(width/2), np.array(dist_year1), width/2, label=fisrtLabel, alpha=0.9, log=True)
  ax.bar(np.arange(len(x_labels)), np.array(dist_year2), width/2, label=secondLabel, alpha=0.9, log=True)
  ax.bar(np.arange(len(x_labels))+(width/2), np.array(dist_year3), width/2, label=ThirdLabel, alpha=0.9, log=True)
  plt.xticks(np.arange(len(x_labels)), x_labels, rotation=45)
  plt.xlabel("Node Degree")
  plt.legend()
  fig.savefig("Deg_Dist.pdf", bbox_inches='tight')

def plot_Feature_Dist(Features1, Features2, Features3, step, fisrtLabel, secondLabel, ThirdLabel, xlabel, intg = 0):
  if(intg):
    Features1 = Features1.astype('int')
    Features2 = Features2.astype('int')
    Features3 = Features3.astype('int')
    delta_year1 = np.max(Features1)//step
    delta_year2 = np.max(Features2)//step
    delta_year3 = np.max(Features3)//step
  else:
    delta_year1 = np.max(Features1)/step
    delta_year2 = np.max(Features2)/step
    delta_year3 = np.max(Features3)/step
  dist_year1 = []
  dist_year2 = []
  dist_year3 = []

  for i in range(int(step*delta_year3/delta_year1)+1):
    dist_year1.append(len(Features1[(Features1<(i+1)*delta_year1) * (Features1>(i)*delta_year1)]))
    dist_year2.append(len(Features2[(Features2<(i+1)*delta_year1) * (Features2>(i)*delta_year1)]))
    dist_year3.append(len(Features3[(Features3<(i+1)*delta_year1) * (Features3>(i)*delta_year1)]))

  dist_year1 = np.array(dist_year1)
  dist_year2 = np.array(dist_year2)
  dist_year3 = np.array(dist_year3)
  sns.set()
  width = 0.35  # the width of the bars
  x_labels_st = np.arange(0,int(step*delta_year3/delta_year1)+1)*(delta_year1)
  if(intg==0):
    x_labels_st = (x_labels_st*10000).astype('int')/10000
  x_labels = np.array([str(x_) for x_ in x_labels_st])
  fig = plt.figure(figsize=(step,3))
  ax = fig.add_axes([0,0,1,1])
  ax.bar(np.arange(len(x_labels))-(width/2), np.array(dist_year1), width/2, label=fisrtLabel, alpha=0.8, log=True)
  ax.bar(np.arange(len(x_labels)), np.array(dist_year2), width/2, label=secondLabel, alpha=0.8, log=True)
  ax.bar(np.arange(len(x_labels))+(width/2), np.array(dist_year3), width/2, label=ThirdLabel, alpha=0.8, log=True)
  plt.xticks(np.arange(len(x_labels)), x_labels, rotation=45)
  plt.xlabel(xlabel)
  plt.legend()
  #plt.savefig(xlabel+".png")
  fig.savefig(xlabel+".pdf", bbox_inches='tight')

def plot_shp_Dist(Features1, Features2, Features3, fisrtLabel, secondLabel, ThirdLabel, xlabel):
  dist_year1 = []
  dist_year2 = []
  dist_year3 = []

  for i in range(10):
    dist_year1.append(len(Features1[Features1==i]))
    dist_year2.append(len(Features2[Features2==i]))
    dist_year3.append(len(Features3[Features3==i]))

  sns.set()
  width = 0.35  # the width of the bars
  x_labels_st = np.arange(10)
  x_labels = np.array([str(x_) for x_ in x_labels_st])
  fig = plt.figure(figsize=(6,3))
  ax = fig.add_axes([0,0,1,1])
  ax.bar(np.arange(10)-(width/2), np.array(dist_year1), width/2, label=fisrtLabel, alpha=0.8, log=True)
  ax.bar(np.arange(10), np.array(dist_year2), width/2, label=secondLabel, alpha=0.8, log=True)
  ax.bar(np.arange(10)+(width/2), np.array(dist_year3), width/2, label=ThirdLabel, alpha=0.8, log=True)
  plt.xticks(np.arange(len(x_labels)), x_labels, rotation=45)
  plt.xlabel(xlabel)
  plt.legend()
  fig.savefig(xlabel+".pdf", bbox_inches='tight')
  
  
def plot_isol_high_Deg(year_start, year_end):
  highDeg = []
  IsolDeg = []
  for i in range(year_start, year_end):
    if(i==2014):
      degs = np.array(graph_2014.degree())[:,1]
      IsolDeg.append(sum(degs==0))
      highDeg.append(max(degs))
    elif(i==2015):
      degs = np.array(graph_2015.degree())[:,1]
      IsolDeg.append(sum(degs==0))
      highDeg.append(max(degs))
    elif(i==2016):
      degs = np.array(graph_2016.degree())[:,1]
      IsolDeg.append(sum(degs==0))
      highDeg.append(max(degs))
    elif(i==2017):
      degs = np.array(graph_2017.degree())[:,1]
      IsolDeg.append(sum(degs==0))
      highDeg.append(max(degs))
    else:
      graph = graph_year(full_dynamic_graph_sparse, i)
      degs = np.array(graph.degree())[:,1]
      IsolDeg.append(sum(degs==0))
      highDeg.append(max(degs))

  x = np.arange(year_start, year_end)
  x_labels = np.array([str(x_) for x_ in x])
  sns.set()
  width = 0.35  # the width of the bars
  fig = plt.figure(figsize=(10,3))
  ax = fig.add_axes([0,0,1,1])
  ax.bar(x-width/2, IsolDeg, width, alpha=0.8, label='Number of isolated nodes')
  #ax.plot(x-width/2, IsolDeg, '-o', alpha=0.7)
  ax.bar(x+width/2, highDeg, width, alpha=0.8, label='highest degree')
  #ax.plot(x+width/2, highDeg, '-o', alpha=0.7)
  plt.xticks(x, x_labels)
  plt.xlabel("year")
  plt.legend()
  fig.savefig("isol_high_Deg.pdf", bbox_inches='tight')

plot_Deg_Dist(graph_2015, graph_2016, graph_2017, 10, "2015", "2016", "2017")
plot_Feature_Dist(cn_2015, cn_2016, cn_2017, 10, "2015", "2016", "2017", xlabel="common neighbors", intg=1)
plot_Feature_Dist(and_2015, and_2016, and_2017, 10, "2015", "2016", "2017", xlabel="average neighbor degree", intg=1)
plot_Feature_Dist(ja_2015, ja_2016, ja_2017, 10, "2015", "2016", "2017", xlabel="ja card")
plot_Feature_Dist(pr_2015, pr_2016, pr_2017, 10, "2015", "2016", "2017", xlabel="page rank")
plot_shp_Dist(shp_2015, shp_2016, shp_2017, "2015", "2016", "2017", xlabel="shortest path")
plot_isol_high_Deg(2010,2018)
