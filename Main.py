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
import xlrd
import openpyxl
import numpy as np
import xlsxwriter
from matplotlib.dates import DateFormatter
import datetime as dt
from pathlib import Path
from collections import Counter
import scipy.stats as ss
from scipy.stats import chi2, chi2_contingency
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from datetime import timedelta, date
import pyreadstat
import prince
import decimal
import seaborn as sn
import statsmodels.formula.api as smf
import statsmodels.api as st
from statsmodels.stats import contingency_tables
from statsmodels.stats.contingency_tables import (mcnemar, cochrans_q, SquareTable)
import sklearn.metrics.pairwise
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import itertools
import statsmodels.formula.api as smf

#from datamodeller import cramers_v, _inf_nan_str, cramers_vc


def is_categorical(array_like):
    return array_like.dtype.name == 'category'

def df_to_plotly(df):
    return {'z': df.values.tolist(),
            'x': df.columns.tolist(),
            'y': df.index.tolist()}

def create_directories(all_names):
    
    for item in all_names:
        out_path_im = Path(cwd, "Data/OutIm/"+item)
        out_path_im.mkdir(parents=True, exist_ok=True)
        out_path_files = Path(cwd, "Data/Out/"+ item)
        out_path_files.mkdir(parents=True, exist_ok=True)
    return 


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
#prepare data frame, unique per data
dataset_I = pd.read_csv(file_open, encoding='latin-1')
dataset_I["SiteName"] = dataset_I["SiteName"].str.lower()
dataset_I["UltimateForm"] = dataset_I["UltimateForm"].str.lower()
all_names= set(dataset_I['UltimateForm'])
create_directories(all_names)
one_site=[]
one_type = []
for item in all_names:
    out_path_files = Path(cwd, "Data/Out/"+item)
    out_path_im = Path(cwd, "Data/OutIm/"+item)
    dataset_I_IM = dataset_I[dataset_I['UltimateForm']==item]
    #remane sites to lower casa
    dataset_I_IM["SiteName"] = dataset_I_IM["SiteName"].str.lower()
    if len(set(dataset_I_IM["SiteName"])) ==1:
        one_site.append(item)
    dataset_I_IM['ArtifactTypeNameGeneral'] = [word[:2].upper() for word in dataset_I_IM.ArtifactTypeName]
    if len(set(dataset_I_IM["ArtifactTypeNameGeneral"])) ==1:
        one_type.append(item)
    ArtifactTypeName = pd.get_dummies(dataset_I_IM.ArtifactTypeName)
    ArtifactTypeNameGeneral = pd.get_dummies(dataset_I_IM.ArtifactTypeNameGeneral)
    dataset_bells_locs = pd.concat([dataset_I_IM.SiteName, dataset_I_IM.Latitude, dataset_I_IM.Longitude], axis=1).groupby(['SiteName']).mean()
    dataset_bells_arts = pd.concat([dataset_I_IM.SiteName,ArtifactTypeNameGeneral, ArtifactTypeName], axis=1).groupby(['SiteName']).sum()
    dataset_bells = pd.concat([dataset_bells_locs,dataset_bells_arts],axis=1)
    name='dataset_'+item+'.csv'
    file_save = out_path_files / name
    dataset_bells.to_csv(file_save, encoding = 'utf-8-sig')
    #scatter plots
    plot_scale = 10 
    
    for i, col_name in enumerate(dataset_bells.columns):
        if len(col_name) > 2 and col_name!="Latitude"and col_name!="Longitude":
            first_long_col = i
     #       if item=="ring":
     #           print(first_long_col)
     #           print(col_name)
            break
        first_long_col= int(dataset_bells.shape[1]/2)-1
            #print(first_long_col)
    if first_long_col==1:
        first_long_col=3
    for colid in range(2,first_long_col):
        fig = go.Figure(data=go.Scatter(
        x=dataset_bells.Longitude, 
        y=dataset_bells.Latitude,
        hovertemplate =
        '<b>%{text}</b>'+
        '<br>Longitude: %{x:.2f}'+
        '<br>Latitude: %{y:.2f}<br>', 
        text=dataset_bells.index.tolist(),
        mode =   'markers',
        marker = dict(
            size = dataset_bells.iloc[:,colid].to_numpy()*plot_scale,
            color = dataset_bells.iloc[:,colid].to_numpy()*plot_scale, #set color equal to a variable
            #color = labels, #set color equal to a variable
            colorscale='Inferno', # one of plotly colorscales
            showscale=True
        )
        )) 
        fig.update_layout(yaxis_range=[dataset_bells.Latitude.min()-1,dataset_bells.Latitude.max()+1])
        fig.update_layout(xaxis_range=[dataset_bells.Longitude.min()-1,dataset_bells.Longitude.max()+1])
        file_save = out_path_im / (item+"_artgeneral_" + dataset_bells.iloc[:,colid].name + ".html")
        fig.write_html(file_save)
    plot_scale = 10
    for colid in range(first_long_col,dataset_bells.shape[1] ):
        #print(dataset_bells.iloc[:,colid].name)
        fig = go.Figure(data=go.Scatter(
            x=dataset_bells.Longitude, 
            y=dataset_bells.Latitude,
            hovertemplate =
            '<b>%{text}</b>'+
            '<br>Longitude: %{x:.2f}'+
            '<br>Latitude: %{y:.2f}<br>', 
            text=dataset_bells.index.tolist(),
            mode =   'markers',
            marker = dict(
                size = dataset_bells.iloc[:,colid].to_numpy()*plot_scale,
                color = dataset_bells.iloc[:,colid].to_numpy()*plot_scale, #set color equal to a variable
                #color = labels, #set color equal to a variable
                colorscale='Inferno', # one of plotly colorscales
                showscale=True
            )
        )) 
        fig.update_layout(yaxis_range=[dataset_bells.Latitude.min()-1,dataset_bells.Latitude.max()+1])
        fig.update_layout(xaxis_range=[dataset_bells.Longitude.min()-1,dataset_bells.Longitude.max()+1])
        file_save = out_path_im / (item+"_art_" + dataset_bells.iloc[:,colid].name + ".html")
        fig.write_html(file_save)
    d = {'col1': [np.zeros(1)]}
    dataset_art = pd.DataFrame(data=d, index=dataset_bells.iloc[:,2:first_long_col].columns.tolist())
    rowid = 0
    for colid in range(2, first_long_col):
        dataset_art.iloc[rowid,0] = ((dataset_bells.iloc[:,colid] != 0).sum()/len(dataset_bells))*100
        rowid = rowid + 1
    file_save = out_path_files / ('dataset_'+item+'_GeneralType.csv')
    dataset_art.to_csv(file_save, encoding = 'utf-8-sig')

    d = {'col1': [np.zeros(1)]}
    dataset_art = pd.DataFrame(data=d, index=dataset_bells.iloc[:,first_long_col:dataset_bells.shape[1]].columns.tolist())
    rowid = 0
    for colid in range(first_long_col, dataset_bells.shape[1]):
        dataset_art.iloc[rowid,0] = ((dataset_bells.iloc[:,colid] != 0).sum()/len(dataset_bells))*100
        rowid = rowid + 1
    file_save = out_path_files / ('dataset_'+item+'_SpecificType.csv')
    dataset_art.to_csv(file_save, encoding = 'utf-8-sig') 

    
    dataset_bells_L = dataset_bells.iloc[:,0:2]
    dataset_bells_G = dataset_bells.iloc[:,2:first_long_col]
    dataset_bells_S = dataset_bells.iloc[:,first_long_col:dataset_bells.shape[1]]


    dataset_bellsbin = dataset_bells.copy()
    a =dataset_bellsbin.iloc[:,2:dataset_bells.shape[1]]
    a[a > 0] = 1
    dataset_bellsbin.iloc[:,2:dataset_bells.shape[1]] = a
    dataset_bells_Lbin = dataset_bellsbin.iloc[:,0:2]
    dataset_bells_Gbin = dataset_bellsbin.iloc[:,2:first_long_col]
    dataset_bells_Sbin = dataset_bellsbin.iloc[:,first_long_col:dataset_bells.shape[1]]


    dataset_bells_G_cs = pd.DataFrame(columns=dataset_bells.index.tolist(), index=dataset_bells.index.tolist(), dtype=np.float32)
    dataset_bells_S_cs = pd.DataFrame(columns=dataset_bells.index.tolist(), index=dataset_bells.index.tolist(), dtype=np.float32)
    N=dataset_bells_Gbin.shape[1]
    M=dataset_bells_Sbin.shape[1]
    if item=="ring":
        dataset_bells_Gbin.to_csv(out_path_gen/"Gbin.csv",encoding = 'utf-8-sig')
        dataset_bells_Sbin.to_csv(out_path_gen/"Sbin.csv",encoding = 'utf-8-sig')
    if item == "bell":
        dataset_bells_Sbin.to_csv(out_path_gen/"bell_Sbin.csv",encoding = 'utf-8-sig')

    for pair in itertools.combinations(dataset_bellsbin.index.tolist(), 2):
        dataset_bells_G_cs.loc[pair[0], pair[1]]= np.count_nonzero((dataset_bells_Gbin.loc[pair[0],:])&(dataset_bells_Gbin.loc[pair[1],:]))/N
        dataset_bells_G_cs.loc[pair[1], pair[0]] = dataset_bells_G_cs.loc[pair[0], pair[1]]
        dataset_bells_S_cs.loc[pair[0], pair[1]]= np.count_nonzero((dataset_bells_Sbin.loc[pair[0],:])&(dataset_bells_Sbin.loc[pair[1],:]))/M
        dataset_bells_S_cs.loc[pair[1], pair[0]] = dataset_bells_S_cs.loc[pair[0], pair[1]]
    if item=="ring":
        dataset_bells_G_cs.to_csv(out_path_gen/"G_csbin.csv",encoding = 'utf-8-sig')
        dataset_bells_S_cs.to_csv(out_path_gen/"S_csbin.csv",encoding = 'utf-8-sig')
    if flag:
        mask = np.triu(np.ones_like(dataset_bells_G_cs, dtype=bool))
        f, ax = plt.subplots(figsize=(11, 9))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(dataset_bells_G_cs, mask=mask, cmap=cmap, center=0.3,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
        file_save = out_path_im / ("CS_G.png")
        plt.savefig(file_save, dpi=900)
        plt.close()

        mask = np.triu(np.ones_like(dataset_bells_S_cs, dtype=bool))
        f, ax = plt.subplots(figsize=(11, 9))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(dataset_bells_S_cs, mask=mask, cmap=cmap, center=0.01,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
        file_save = out_path_im / ("CS_S.png")
        plt.savefig(file_save, dpi=900)
        plt.close()

    dataset_bells_G_csb = dataset_bells_G_cs.copy()
    dataset_bells_S_csb = dataset_bells_S_cs.copy()
    dataset_bells_G_csb[dataset_bells_G_csb>0]=1
    dataset_bells_S_csb[dataset_bells_S_csb>0]=1
    dataset_bells_G_csb = dataset_bells_G_csb + dataset_bells_S_csb
    dataset_bells_G_csb[dataset_bells_G_csb==2] = 10
    conections_totals = dataset_bells_G_cs+dataset_bells_S_cs
    conections_totals = conections_totals/conections_totals.max().max()
    
    nodes = pd.DataFrame(data=np.count_nonzero(dataset_bells_Sbin, axis =1), index=dataset_bells_Sbin.index)
    nodes = nodes.reset_index().reset_index()
    nodes.columns = ['ID','label', 'weight']
    file_save = out_path_files / (item+'_nodesG.csv')
    nodes.to_csv(file_save, float_format="%.4f", index=False, encoding='utf-8-sig') 

    if item=="ring":
        print(nodes)

    dataset_bells_G_csb = dataset_bells_G_csb.reset_index(drop=True)
    dataset_bells_G_csb.columns = range(dataset_bells_G_csb.columns.size)
    dataset_bells_G_csb = dataset_bells_G_csb.mask(np.triu(np.ones_like(dataset_bells_G_csb, dtype=bool),0))
    dataset_bells_G_csb = dataset_bells_G_csb.unstack()
    dataset_bells_G_csb = dataset_bells_G_csb[~dataset_bells_G_csb.isin([0, np.nan, np.inf, -np.inf])]
    dataset_bells_G_csb = dataset_bells_G_csb.to_frame()
    dataset_bells_G_csb = dataset_bells_G_csb.reset_index()
    dataset_bells_G_csb.rename(columns={'level_0': 'Source', 'level_1': 'Target', 0: 'weight'}, inplace=True)
    file_save = out_path_files / (item+'_edgesGB.csv')
    dataset_bells_G_csb.to_csv(file_save, float_format="%.4f", index=False) 
    #aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    if item=="ring":
        print(dataset_bells_G_csb)

    conections_totals = conections_totals.reset_index(drop=True)
    conections_totals.columns = range(conections_totals.columns.size)
    conections_totals = conections_totals.mask(np.triu(np.ones_like(conections_totals, dtype=bool),0))
    conections_totals = conections_totals.unstack()
    conections_totals = conections_totals[~conections_totals.isin([0, np.nan, np.inf, -np.inf])]
    conections_totals = conections_totals.to_frame()
    conections_totals = conections_totals.reset_index()
    conections_totals.rename(columns={'level_0': 'Source', 'level_1': 'Target', 0: 'weight'}, inplace=True)
    file_save = out_path_files / (item+'_edges.csv')
    conections_totals.to_csv(file_save, float_format="%.4f", index=False) 


    dataset_bells_S_cs = pd.DataFrame(columns=dataset_bells.index.tolist(), index=dataset_bells.index.tolist(), dtype=np.float32)
    dataset_bells_Sbinn=dataset_bells_Sbin
    dataset_bells_Sbinn = dataset_bells_Sbinn.drop(dataset_bells_Sbinn.columns[0], axis = 1)
    if item=="ring":
   #     print(dataset_bells_S_cs)
        print(dataset_bells_Sbinn)

    M=dataset_bells_Sbinn.shape[1]
    if M==0:
        M=1
    for pair in itertools.combinations(dataset_bells_Sbinn.index.tolist(), 2):
        dataset_bells_S_cs.loc[pair[0], pair[1]]= np.count_nonzero((dataset_bells_Sbinn.loc[pair[0],:])&(dataset_bells_Sbinn.loc[pair[1],:]))/M
        dataset_bells_S_cs.loc[pair[1], pair[0]] = dataset_bells_S_cs.loc[pair[0], pair[1]]

    conections_totals = dataset_bells_S_cs
    conections_totals = conections_totals/conections_totals.max().max()
    conections_totals = conections_totals.reset_index(drop=True)
    conections_totals.columns = range(conections_totals.columns.size)
    conections_totals = conections_totals.mask(np.triu(np.ones_like(conections_totals, dtype=bool),0))
    conections_totals = conections_totals.unstack()
    conections_totals = conections_totals[~conections_totals.isin([0, np.nan, np.inf, -np.inf])]
    conections_totals = conections_totals.to_frame()
    conections_totals = conections_totals.reset_index()
    conections_totals.rename(columns={'level_0': 'Source', 'level_1': 'Target', 0: 'weight'}, inplace=True)
    file_save = out_path_files / (item+'_edgesSnA.csv')
    conections_totals.to_csv(file_save, float_format="%.4f", index=False)

#print(one_site)
#print(one_type)
