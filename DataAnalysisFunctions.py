from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
def is_categorical(array_like):
    return array_like.dtype.name == 'category'

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
            if artifact_name=="ring":
                print(first_long_col)
                print(col_name)
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