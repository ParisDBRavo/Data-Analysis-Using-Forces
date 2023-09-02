from pathlib import Path
import pandas as pd
import Tools
import numpy as np
def forceConstantRepelent(pair, dataset_I, flag=False):
    #New part to take real distance into account but it is not yet implemented
    """if flag:
        cwd = Path.cwd()
        file_I = Path("Data/In/copper_29_11mod.csv")
        file_open = cwd / file_I
        dataset_O = pd.read_csv(file_open, encoding='latin-1')
        print(dataset_O['SiteName'] == pair[0])
        print(dataset_O.loc[ dataset_O['SiteName'] == pair[0]])
        print(dataset_O.loc[ dataset_O['SiteName'] == pair[1],'Latitude'])"""
    lengthOfBellTypes=dataset_I.shape[1]-1
    firstSiteRowNumber = dataset_I.loc[dataset_I["SiteName"]==pair[0]].index[0]
    secondSiteRowNumber = dataset_I.loc[dataset_I["SiteName"]==pair[1]].index[0]
    dataset_I = dataset_I.drop(dataset_I.columns[0], axis = 1)
    numberOfBellsInCommon= np.count_nonzero((dataset_I.loc[firstSiteRowNumber,:].astype(int))&(dataset_I.loc[secondSiteRowNumber,:].astype(int)))/lengthOfBellTypes
    if numberOfBellsInCommon!=0:
        return numberOfBellsInCommon
    else:
        return -0.1
    
def forceReducingZeroesConstantRepelent(pair, dataset_I):
    lengthOfBellTypes=dataset_I.shape[1]-1
    firstRowNumber = dataset_I.loc[dataset_I["SiteName"]==pair[0]].index[0]
    secondRowNumber = dataset_I.loc[dataset_I["SiteName"]==pair[1]].index[0]
    dataset_I = dataset_I.drop(dataset_I.columns[0], axis = 1)
    firstSiteRow, secondSiteRow =Tools.deleteZeroesFromBoth(dataset_I.loc[firstRowNumber,:].astype(int),dataset_I.loc[secondRowNumber,:].astype(int))
    lengthOfBellTypes=len(firstSiteRow)
    numberOfBellsInCommon= np.count_nonzero((firstSiteRow)&(secondSiteRow))/lengthOfBellTypes
    #First Try
    if numberOfBellsInCommon!=0:
        return numberOfBellsInCommon
    else:
        return -0.1
    
#Los sitios se atraen si tienen por lo menos un cascabel en común
# la fuerza de atracción es igual al número de cascabeles que tienen en común, 
# se repelen de acuerdo con el número total que no tienen en común
def forceRepelentNonequal(pair, dataset_I, flag=False):
    lengthOfBellTypes=dataset_I.shape[1]-1
    firstRowNumber = dataset_I.loc[dataset_I["SiteName"]==pair[0]].index[0]
    secondRowNumber = dataset_I.loc[dataset_I["SiteName"]==pair[1]].index[0]
    dataset_I = dataset_I.drop(dataset_I.columns[0], axis = 1)
    numberOfBellsInCommon= np.count_nonzero((dataset_I.loc[firstRowNumber,:].astype(int))&(dataset_I.loc[secondRowNumber,:].astype(int)))/lengthOfBellTypes
    if numberOfBellsInCommon!=0:
        return numberOfBellsInCommon
    else:
        return numberOfBellsInCommon-lengthOfBellTypes
#Los sitios se atraen si tienen por lo menos un cascabel en común, 
# la fuerza es igual para todos si dos sitios tienen en común algo se atraen con fuerza x y 
# si no tienen nada en común se repelen con fuerza y, donde x y y son iguales
    
def forceAtractionEqualRepelent(pair, dataset_I, flag=False):
    lengthOfBellTypes=dataset_I.shape[1]-1
    firstRowNumber = dataset_I.loc[dataset_I["SiteName"]==pair[0]].index[0]
    secondRowNumber = dataset_I.loc[dataset_I["SiteName"]==pair[1]].index[0]
    dataset_I = dataset_I.drop(dataset_I.columns[0], axis = 1)
    numberOfBellsInCommon= np.count_nonzero((dataset_I.loc[firstRowNumber,:].astype(int))&(dataset_I.loc[secondRowNumber,:].astype(int)))/lengthOfBellTypes
    if numberOfBellsInCommon!=0:
        return 1.0
    else:
        return -1.0
#Los sitios se atraen con respecto a si tienen o no un cascabel en común, 
# la masa del sitio es el número total de tipos de cascabeles que tiene el sitio, 
# con que tengan uno en común se atrae si no tienen nada en común se repelen  
def forceSameGravity(pair, dataset_I, flag=False):
    lengthOfBellTypes=dataset_I.shape[1]-1
    firstRowNumber = dataset_I.loc[dataset_I["SiteName"]==pair[0]].index[0]
    secondRowNumber = dataset_I.loc[dataset_I["SiteName"]==pair[1]].index[0]
    dataset_I = dataset_I.drop(dataset_I.columns[0], axis = 1)
    numberOfBellsInCommon= np.count_nonzero((dataset_I.loc[firstRowNumber,:].astype(int))&(dataset_I.loc[secondRowNumber,:].astype(int)))/lengthOfBellTypes
    massFirstSite =np.count_nonzero(dataset_I.loc[firstRowNumber,:].astype(int))
    massSecondSite=np.count_nonzero(dataset_I.loc[secondRowNumber,:].astype(int))
    if numberOfBellsInCommon!=0:
        return massFirstSite*massSecondSite
    else:
        return -massFirstSite*massSecondSite
    
