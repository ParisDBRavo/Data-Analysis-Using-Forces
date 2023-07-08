import os
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.express as px
import plotly.io as pio
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

#from datamodeller import cramers_v, _inf_nan_str, cramers_vc

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
one_site=[]
one_type = []
for artifact_name in all_names:
    out_path_files = Path(cwd, "Data/Out/"+artifact_name)
    out_path_im = Path(cwd, "Data/OutIm/"+artifact_name)
    dataset_I_IM = dataset_I[dataset_I['UltimateForm']==artifact_name]
    #remane sites to lower case
    dataset_I_IM["SiteName"] = dataset_I_IM["SiteName"].str.lower()
    if len(set(dataset_I_IM["SiteName"])) ==1:
        one_site.append(artifact_name)
    dataset_I_IM['ArtifactTypeNameGeneral'] = [word[:2].upper() for word in dataset_I_IM.ArtifactTypeName]
    if len(set(dataset_I_IM["ArtifactTypeNameGeneral"])) ==1:
        one_type.append(artifact_name)
    ArtifactTypeName = pd.get_dummies(dataset_I_IM.ArtifactTypeName)
    ArtifactTypeNameGeneral = pd.get_dummies(dataset_I_IM.ArtifactTypeNameGeneral)
    dataset_locs = pd.concat([dataset_I_IM.SiteName, dataset_I_IM.Latitude, dataset_I_IM.Longitude], axis=1).groupby(['SiteName']).mean()
    dataset_arts = pd.concat([dataset_I_IM.SiteName,ArtifactTypeNameGeneral, ArtifactTypeName], axis=1).groupby(['SiteName']).sum()
    dataset = pd.concat([dataset_locs,dataset_arts],axis=1)
    name='dataset_'+artifact_name+'.csv'
    file_save = out_path_files / name
    dataset.to_csv(file_save, encoding = 'utf-8-sig')
    #scatter plots
    plot_scale = 10 
    first_specific_type= DataAnalysisFunctions.get_first_specific_type(dataset,artifact_name)
    for colid in range(2,first_specific_type):
        fig = go.Figure(data=go.Scatter(
        x=dataset.Longitude, 
        y=dataset.Latitude,
        hovertemplate =
        '<b>%{text}</b>'+
        '<br>Longitude: %{x:.2f}'+
        '<br>Latitude: %{y:.2f}<br>', 
        text=dataset.index.tolist(),
        mode =   'markers',
        marker = dict(
            size = dataset.iloc[:,colid].to_numpy()*plot_scale,
            color = dataset.iloc[:,colid].to_numpy()*plot_scale, #set color equal to a variable
            #color = labels, #set color equal to a variable
            colorscale='Inferno', # one of plotly colorscales
            showscale=True
        )
        )) 
        fig.update_layout(yaxis_range=[dataset.Latitude.min()-1,dataset.Latitude.max()+1])
        fig.update_layout(xaxis_range=[dataset.Longitude.min()-1,dataset.Longitude.max()+1])
        file_save = out_path_im / (artifact_name+"_artgeneral_" + dataset.iloc[:,colid].name + ".html")
        fig.write_html(file_save)
    plot_scale = 10
    for colid in range(first_specific_type,dataset.shape[1] ):
        #print(dataset_bells.iloc[:,colid].name)
        fig = go.Figure(data=go.Scatter(
            x=dataset.Longitude, 
            y=dataset.Latitude,
            hovertemplate =
            '<b>%{text}</b>'+
            '<br>Longitude: %{x:.2f}'+
            '<br>Latitude: %{y:.2f}<br>', 
            text=dataset.index.tolist(),
            mode =   'markers',
            marker = dict(
                size = dataset.iloc[:,colid].to_numpy()*plot_scale,
                color = dataset.iloc[:,colid].to_numpy()*plot_scale, #set color equal to a variable
                #color = labels, #set color equal to a variable
                colorscale='Inferno', # one of plotly colorscales
                showscale=True
            )
        )) 
        fig.update_layout(yaxis_range=[dataset.Latitude.min()-1,dataset.Latitude.max()+1])
        fig.update_layout(xaxis_range=[dataset.Longitude.min()-1,dataset.Longitude.max()+1])
        file_save = out_path_im / (artifact_name+"_art_" + dataset.iloc[:,colid].name + ".html")
        fig.write_html(file_save)
    d = {'col1': [np.zeros(1)]}
    dataset_art = pd.DataFrame(data=d, index=dataset.iloc[:,2:first_specific_type].columns.tolist())
    rowid = 0
    for colid in range(2, first_specific_type):
        dataset_art.iloc[rowid,0] = ((dataset.iloc[:,colid] != 0).sum()/len(dataset))*100
        rowid = rowid + 1
    file_save = out_path_files / ('dataset_'+artifact_name+'_GeneralType.csv')
    dataset_art.to_csv(file_save, encoding = 'utf-8-sig')

    d = {'col1': [np.zeros(1)]}
    dataset_art = pd.DataFrame(data=d, index=dataset.iloc[:,first_specific_type:dataset.shape[1]].columns.tolist())
    rowid = 0
    for colid in range(first_specific_type, dataset.shape[1]):
        dataset_art.iloc[rowid,0] = ((dataset.iloc[:,colid] != 0).sum()/len(dataset))*100
        rowid = rowid + 1
    file_save = out_path_files / ('dataset_'+artifact_name+'_SpecificType.csv')
    dataset_art.to_csv(file_save, encoding = 'utf-8-sig') 

    
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


    dataset_G_cs = pd.DataFrame(columns=dataset.index.tolist(), index=dataset.index.tolist(), dtype=np.float32)
    dataset_S_cs = pd.DataFrame(columns=dataset.index.tolist(), index=dataset.index.tolist(), dtype=np.float32)
    N=dataset_Gbin.shape[1]
    M=dataset_Sbin.shape[1]
    if artifact_name=="ring":
        dataset_Gbin.to_csv(out_path_gen/"Gbin.csv",encoding = 'utf-8-sig')
        dataset_Sbin.to_csv(out_path_gen/"Sbin.csv",encoding = 'utf-8-sig')
    if artifact_name == "bell":
        dataset_Sbin.to_csv(out_path_gen/"bell_Sbin.csv",encoding = 'utf-8-sig')

    for pair in itertools.combinations(dataset_bin.index.tolist(), 2):
        dataset_G_cs.loc[pair[0], pair[1]]= np.count_nonzero((dataset_Gbin.loc[pair[0],:])&(dataset_Gbin.loc[pair[1],:]))/N
        dataset_G_cs.loc[pair[1], pair[0]] = dataset_G_cs.loc[pair[0], pair[1]]
        dataset_S_cs.loc[pair[0], pair[1]]= np.count_nonzero((dataset_Sbin.loc[pair[0],:])&(dataset_Sbin.loc[pair[1],:]))/M
        dataset_S_cs.loc[pair[1], pair[0]] = dataset_S_cs.loc[pair[0], pair[1]]
    if artifact_name=="ring":
        dataset_G_cs.to_csv(out_path_gen/"G_csbin.csv",encoding = 'utf-8-sig')
        dataset_S_cs.to_csv(out_path_gen/"S_csbin.csv",encoding = 'utf-8-sig')
    if flag:
        DataAnalysisFunctions.save_images_heatmaps(dataset_G_cs,dataset_S_cs, out_path_im)

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
    file_save = out_path_files / (artifact_name+'_nodesG.csv')
    nodes.to_csv(file_save, float_format="%.4f", index=False, encoding='utf-8-sig') 

    if artifact_name=="ring":
        print(nodes)

    dataset_G_csb = dataset_G_csb.reset_index(drop=True)
    dataset_G_csb.columns = range(dataset_G_csb.columns.size)
    dataset_G_csb = dataset_G_csb.mask(np.triu(np.ones_like(dataset_G_csb, dtype=bool),0))
    dataset_G_csb = dataset_G_csb.unstack()
    dataset_G_csb = dataset_G_csb[~dataset_G_csb.isin([0, np.nan, np.inf, -np.inf])]
    dataset_G_csb = dataset_G_csb.to_frame()
    dataset_G_csb = dataset_G_csb.reset_index()
    dataset_G_csb.rename(columns={'level_0': 'Source', 'level_1': 'Target', 0: 'weight'}, inplace=True)
    file_save = out_path_files / (artifact_name+'_edgesGB.csv')
    dataset_G_csb.to_csv(file_save, float_format="%.4f", index=False) 
    if artifact_name=="ring":
        print(dataset_G_csb)

    conections_totals = conections_totals.reset_index(drop=True)
    conections_totals.columns = range(conections_totals.columns.size)
    conections_totals = conections_totals.mask(np.triu(np.ones_like(conections_totals, dtype=bool),0))
    conections_totals = conections_totals.unstack()
    conections_totals = conections_totals[~conections_totals.isin([0, np.nan, np.inf, -np.inf])]
    conections_totals = conections_totals.to_frame()
    conections_totals = conections_totals.reset_index()
    conections_totals.rename(columns={'level_0': 'Source', 'level_1': 'Target', 0: 'weight'}, inplace=True)
    file_save = out_path_files / (artifact_name+'_edges.csv')
    conections_totals.to_csv(file_save, float_format="%.4f", index=False) 


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
    file_save = out_path_files / (artifact_name+'_edgesSnA.csv')
    conections_totals.to_csv(file_save, float_format="%.4f", index=False)