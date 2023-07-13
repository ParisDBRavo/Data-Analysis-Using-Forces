import plotly.figure_factory as ff
import plotly.graph_objects as go
import pandas as pd
import DataAnalysisFunctions
import numpy as np
from matplotlib.dates import DateFormatter
import datetime as dt
from pathlib import Path
from collections import Counter
from scipy.stats import chi2, chi2_contingency
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import decimal
from statsmodels.stats.contingency_tables import (mcnemar, cochrans_q, SquareTable)
from sklearn.metrics.pairwise import cosine_similarity
import itertools
import statsmodels.formula.api as smf

# create a new context for the decimal format
ctx = decimal.Context()
ctx.prec = 4
prob = 0.95
alpha = 1.0 - prob
#Flag to not calculate figures every run False does not calculate over again
flag=False
cwd = Path.cwd()
file_I = Path("Data/In/copper_29_11mod.csv")
file_open = cwd / file_I
out_path_gen = Path(cwd, "Data/OutGeneralInfo/")
out_path_gen.mkdir(parents=True, exist_ok=True)
dataset_I=DataAnalysisFunctions.prepare_dataset(file_open)
all_names= set(dataset_I['UltimateForm'])
DataAnalysisFunctions.create_directories(all_names, cwd)
for artifact_name in all_names:
    out_path_im = Path(cwd, "Data/OutIm/"+artifact_name)
    dataset = DataAnalysisFunctions.get_clean_dataset(dataset_I,artifact_name)
    DataAnalysisFunctions.savecsv(dataset,'dataset_'+artifact_name+'.csv',artifact_name)
    #scatter plots    
    first_specific_type= DataAnalysisFunctions.get_first_specific_type(dataset,artifact_name)
    DataAnalysisFunctions.printing_images_general_and_specific(dataset, artifact_name)
    #Sending info to a csv
    DataAnalysisFunctions.percentage_of_types_located_in_sites(dataset, artifact_name)

    dataset_L = dataset.iloc[:,0:2]
    dataset_G = dataset.iloc[:,2:first_specific_type]
    dataset_S = dataset.iloc[:,first_specific_type:dataset.shape[1]]

    dataset_bin = dataset.copy()
    a =dataset_bin.iloc[:,2:dataset.shape[1]]
    a[a > 0] = 1
    dataset_bin.iloc[:,2:dataset.shape[1]] = a
    dataset_Lbin = dataset_bin.iloc[:,0:2]
    dataset_Gbin = dataset_bin.iloc[:,2:first_specific_type]
    dataset_Sbin = dataset_bin.iloc[:,first_specific_type:dataset.shape[1]]

    if flag:
        DataAnalysisFunctions.save_images_heatmaps(dataset_G_cs,dataset_S_cs, out_path_im)
    dataset_G_cs,dataset_S_cs=DataAnalysisFunctions.get_dataset_g_and_s_sharing_objects_between_sites(dataset_Gbin,dataset_Sbin,dataset,dataset_bin)
    dataset_G_csb = dataset_G_cs.copy()
    dataset_S_csb = dataset_S_cs.copy()
    dataset_G_csb[dataset_G_csb>0]=1
    dataset_S_csb[dataset_S_csb>0]=1
    dataset_G_csb = dataset_G_csb + dataset_S_csb
    dataset_G_csb[dataset_G_csb==2] = 10
    conections_totals = dataset_G_cs+dataset_S_cs
    conections_totals = conections_totals/conections_totals.max().max()
    nodes = pd.DataFrame(data=np.count_nonzero(dataset_Sbin, axis =1), index=dataset_Sbin.index)
    nodes = nodes.reset_index().reset_index()
    nodes.columns = ['ID','label', 'weight']
    DataAnalysisFunctions.savecsv(nodes,(artifact_name+'_nodesG.csv'), artifact_name)
    
    dataset_G_csb=DataAnalysisFunctions.save_and_calculate_general_edges(artifact_name,dataset_G_csb)

    conections_totals = conections_totals.reset_index(drop=True)
    conections_totals.columns = range(conections_totals.columns.size)
    conections_totals = conections_totals.mask(np.triu(np.ones_like(conections_totals, dtype=bool),0))
    conections_totals = conections_totals.unstack()
    conections_totals = conections_totals[~conections_totals.isin([0, np.nan, np.inf, -np.inf])]
    conections_totals = conections_totals.to_frame()
    conections_totals = conections_totals.reset_index()
    conections_totals.rename(columns={'level_0': 'Source', 'level_1': 'Target', 0: 'weight'}, inplace=True)
    DataAnalysisFunctions.savecsv(conections_totals,(artifact_name+'_edges.csv'),artifact_name,True)

    dataset_S_cs = pd.DataFrame(columns=dataset.index.tolist(), index=dataset.index.tolist(), dtype=np.float32)
    dataset_Sbinn=dataset_Sbin
    dataset_Sbinn = dataset_Sbinn.drop(dataset_Sbinn.columns[0], axis = 1)
    
    if artifact_name=="ring":
        print(dataset_Sbinn)

    M=dataset_Sbinn.shape[1]
    if M==0:
        M=1
    for pair in itertools.combinations(dataset_Sbinn.index.tolist(), 2):
        dataset_S_cs.loc[pair[0], pair[1]]= np.count_nonzero((dataset_Sbinn.loc[pair[0],:])&(dataset_Sbinn.loc[pair[1],:]))/M
        dataset_S_cs.loc[pair[1], pair[0]] = dataset_S_cs.loc[pair[0], pair[1]]

    conections_totals = dataset_S_cs
    conections_totals = conections_totals/conections_totals.max().max()
    conections_totals = conections_totals.reset_index(drop=True)
    conections_totals.columns = range(conections_totals.columns.size)
    conections_totals = conections_totals.mask(np.triu(np.ones_like(conections_totals, dtype=bool),0))
    conections_totals = conections_totals.unstack()
    conections_totals = conections_totals[~conections_totals.isin([0, np.nan, np.inf, -np.inf])]
    conections_totals = conections_totals.to_frame()
    conections_totals = conections_totals.reset_index()
    conections_totals.rename(columns={'level_0': 'Source', 'level_1': 'Target', 0: 'weight'}, inplace=True)
    DataAnalysisFunctions.savecsv(conections_totals,(artifact_name+'_edgesSnA.csv'),artifact_name,True)