from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import itertools
def is_categorical(array_like):
    return array_like.dtype.name == 'category'

def savecsv(dataset,name,artifact_name, flag=False):
    cwd = Path.cwd()
    out_path_files = Path(cwd, "Data/Out/"+artifact_name)
    file_save = out_path_files / name
    if flag:
        dataset.to_csv(file_save, float_format="%.4f", index=False)
    else:
        dataset.to_csv(file_save, encoding = 'utf-8-sig')

def df_to_plotly(df):
    return {'z': df.values.tolist(),
            'x': df.columns.tolist(),
            'y': df.index.tolist()}

def create_directories(all_names, cwd):
    
    for item in all_names:
        out_path_im = Path(cwd, "Data/OutIm/"+item)
        out_path_im.mkdir(parents=True, exist_ok=True)
        out_path_files = Path(cwd, "Data/Out/"+ item)
        out_path_files.mkdir(parents=True, exist_ok=True)
    return 
def prepare_dataset(file_open):
    #prepare data frame, unique per data
    dataset_I = pd.read_csv(file_open, encoding='latin-1')
    dataset_I["SiteName"] = dataset_I["SiteName"].str.lower()
    dataset_I["UltimateForm"] = dataset_I["UltimateForm"].str.lower()  
    return dataset_I 

#this function gives me the first specific type in the dataset
def get_first_specific_type(dataset,artifact_name):
    for i, col_name in enumerate(dataset.columns):
        if len(col_name) > 2 and col_name!="Latitude"and col_name!="Longitude":
            first_long_col = i
            break
        first_long_col= int(dataset.shape[1]/2)-1
    if first_long_col==1:
        first_long_col=3
    return first_long_col

def save_images_heatmaps(dataset_G_cs,dataset_S_cs, out_path_im):
    mask = np.triu(np.ones_like(dataset_G_cs, dtype=bool))
    f, ax = plt.subplots(figsize=(11, 9))
        
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(dataset_G_cs, mask=mask, cmap=cmap, center=0.3,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    file_save = out_path_im / ("CS_G.png")
    plt.savefig(file_save, dpi=900)
    plt.close()

    mask = np.triu(np.ones_like(dataset_S_cs, dtype=bool))
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
        # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(dataset_S_cs, mask=mask, cmap=cmap, center=0.01,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    file_save = out_path_im / ("CS_S.png")
    plt.savefig(file_save, dpi=900)
    plt.close()

def get_clean_dataset(dataset_I, artifact_name):
    one_site=[]
    one_type = []
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
    return dataset

def printing_images_general_and_specific(dataset, artifact_name):
    cwd = Path.cwd()
    out_path_im = Path(cwd, "Data/OutIm/"+artifact_name)
    first_specific_type= get_first_specific_type(dataset,artifact_name)
    plot_scale = 10 
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

def percentage_of_types_located_in_sites(dataset, artifact_name):
    first_specific_type= get_first_specific_type(dataset,artifact_name)
    d = {'col1': [np.zeros(1)]}
    dataset_art = pd.DataFrame(data=d, index=dataset.iloc[:,2:first_specific_type].columns.tolist())
    rowid = 0
    for colid in range(2, first_specific_type):
        dataset_art.iloc[rowid,0] = ((dataset.iloc[:,colid] != 0).sum()/len(dataset))*100
        rowid = rowid + 1
    savecsv(dataset_art, ('dataset_'+artifact_name+'_GeneralType.csv'),artifact_name)

    d = {'col1': [np.zeros(1)]}
    dataset_art = pd.DataFrame(data=d, index=dataset.iloc[:,first_specific_type:dataset.shape[1]].columns.tolist())
    rowid = 0
    for colid in range(first_specific_type, dataset.shape[1]):
        dataset_art.iloc[rowid,0] = ((dataset.iloc[:,colid] != 0).sum()/len(dataset))*100
        rowid = rowid + 1
    savecsv(dataset_art,('dataset_'+artifact_name+'_SpecificType.csv'), artifact_name)

def get_dataset_g_and_s_sharing_objects_between_sites(dataset_Gbin,dataset_Sbin,dataset,dataset_bin):
    dataset_G_cs = pd.DataFrame(columns=dataset.index.tolist(), index=dataset.index.tolist(), dtype=np.float32)
    dataset_S_cs = pd.DataFrame(columns=dataset.index.tolist(), index=dataset.index.tolist(), dtype=np.float32)
    N=dataset_Gbin.shape[1]
    M=dataset_Sbin.shape[1]
    for pair in itertools.combinations(dataset_bin.index.tolist(), 2):
        dataset_G_cs.loc[pair[0], pair[1]]= np.count_nonzero((dataset_Gbin.loc[pair[0],:])&(dataset_Gbin.loc[pair[1],:]))/N
        dataset_G_cs.loc[pair[1], pair[0]] = dataset_G_cs.loc[pair[0], pair[1]]
        dataset_S_cs.loc[pair[0], pair[1]]= np.count_nonzero((dataset_Sbin.loc[pair[0],:])&(dataset_Sbin.loc[pair[1],:]))/M
        dataset_S_cs.loc[pair[1], pair[0]] = dataset_S_cs.loc[pair[0], pair[1]]
    
    return dataset_G_cs,dataset_S_cs
def save_general_edges(artifact_name,dataset_G_csb):
    dataset_G_csb = dataset_G_csb.reset_index(drop=True)
    dataset_G_csb.columns = range(dataset_G_csb.columns.size)
    dataset_G_csb = dataset_G_csb.mask(np.triu(np.ones_like(dataset_G_csb, dtype=bool),0))
    dataset_G_csb = dataset_G_csb.unstack()
    dataset_G_csb = dataset_G_csb[~dataset_G_csb.isin([0, np.nan, np.inf, -np.inf])]
    dataset_G_csb = dataset_G_csb.to_frame()
    dataset_G_csb = dataset_G_csb.reset_index()
    dataset_G_csb.rename(columns={'level_0': 'Source', 'level_1': 'Target', 0: 'weight'}, inplace=True)
    savecsv(dataset_G_csb,(artifact_name+'_edgesGB.csv'), artifact_name)
    return dataset_G_csb