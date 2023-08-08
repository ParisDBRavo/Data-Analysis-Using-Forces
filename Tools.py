import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import itertools
def createRandomPoints(siteNames):
    points ={}
    nums = np.random.choice(range(-1,1+1), size=(1, 2), replace=False) 
    for item in siteNames:
        nums = np.random.uniform(low=-100, high=100, size=(1, 2))
        # Keep only the points that fall inside the circle of radius 1
        #distances = np.sqrt(np.sum(points**2, axis=1))
        #points = points[distances <= 1]
        points[item] = np.asarray((nums[0][0], nums[0][1]))
    return points
def creatingNewDataset(original_df):
    # Extracting the first column
    first_column = original_df.iloc[:, 0]
# Creating a new DataFrame with the first column elements as both column and index
    new_df = pd.DataFrame(index=first_column, columns=first_column)

# Setting diagonal elements to zero using numpy
    np.fill_diagonal(new_df.values, 0)
    return new_df

def gettingLastDistances(LocationOfSites,lastPositionDataSet):
    cwd = Path.cwd()
    out_path_gen = Path(cwd, "Data/BellsOut/")
    out_path_gen.mkdir(parents=True, exist_ok=True)
    out_path_files = Path(cwd, "Data/BellsOut/DistanceData.csv")
    file_save = out_path_files 
    for pair in itertools.combinations(lastPositionDataSet.index, 2):
        x1,y1=LocationOfSites[pair[0]]
        x2,y2=LocationOfSites[pair[1]]
        lastPositionDataSet[pair[0]][pair[1]]=calculateDistanceBetweenTwoPoints(LocationOfSites[pair[0]],LocationOfSites[pair[1]])
        lastPositionDataSet[pair[1]][pair[0]]=(lastPositionDataSet[pair[0]][pair[1]])
    print(lastPositionDataSet)
    lastPositionDataSet.to_csv(file_save, encoding = 'utf-8-sig')
def printingImagesWithNames(points):
    # Create empty lists for x and y coordinates
    x = []
    y = []
    for siteNames in points:
        # Append x and y coordinates to lists
        x.append(points[siteNames][0])
        y.append(points[siteNames][1])
    plt.scatter(x, y)
    for i, siteNames in enumerate(points):
        # Access individual x and y coordinates
        x_coord = points[siteNames][0]
        y_coord = points[siteNames][1]
        # Annotate point with site name
        plt.annotate(siteNames, (x_coord, y_coord))
    plt.show()
#Getting constants so initial velocity equals 0
def gettingConstants(points):
    X0=[]
    Y0=[]
  #  print ("Length points:",len(points))
    for siteNames in points:
        # Append x and y coordinates to lists
        X0.append(points[siteNames][0])
        Y0.append(points[siteNames][1])
    return X0,Y0

def calculateDistanceBetweenTwoPoints(firstPoint, secondPoint):
    return np.linalg.norm(firstPoint - secondPoint)
def calculateDirectionBetweenTwoPoints(firstPoint, secondPoint):
    return firstPoint-secondPoint
def deleteZeroesFromBoth(firstArray,SecondArray):
    mask = np.logical_or(firstArray, SecondArray)
    indices = [i for i, x in enumerate(mask) if x]
    return np.array([firstArray[i] for i in indices]),  np.array([SecondArray[i] for i in indices])

def calculateForceMagnitud(pair, dataset_I):
    M=dataset_I.shape[1]-1
    firstRowNumber = dataset_I.loc[dataset_I["SiteName"]==pair[0]].index[0]
    secondRowNumber = dataset_I.loc[dataset_I["SiteName"]==pair[1]].index[0]
    dataset_I = dataset_I.drop(dataset_I.columns[0], axis = 1)
    firstSiteRow, secondSiteRow =deleteZeroesFromBoth(dataset_I.loc[firstRowNumber,:].astype(int),dataset_I.loc[secondRowNumber,:].astype(int))
    M=len(firstSiteRow)
    number= np.count_nonzero((firstSiteRow)&(secondSiteRow))/M
    #First Try
    if number!=0:
        return number
    else:
        return -0.1
    #if number>=M/2:
    #    return number
    #else:
    #    return number-0.5
def calculateForceMagnitud1(pair, dataset_I, flag=False):
    #New part to take real distance into account but it is not yet implemented
    if flag:
        cwd = Path.cwd()
        file_I = Path("Data/In/copper_29_11mod.csv")
        file_open = cwd / file_I
        dataset_O = pd.read_csv(file_open, encoding='latin-1')
        print(dataset_O['SiteName'] == pair[0])
        print(dataset_O.loc[ dataset_O['SiteName'] == pair[0]])
        print(dataset_O.loc[ dataset_O['SiteName'] == pair[1],'Latitude'])
    M=dataset_I.shape[1]-1
    firstRowNumber = dataset_I.loc[dataset_I["SiteName"]==pair[0]].index[0]
    secondRowNumber = dataset_I.loc[dataset_I["SiteName"]==pair[1]].index[0]
    dataset_I = dataset_I.drop(dataset_I.columns[0], axis = 1)
    number= np.count_nonzero((dataset_I.loc[firstRowNumber,:].astype(int))&(dataset_I.loc[secondRowNumber,:].astype(int)))/M
    if number!=0:
        return number
    else:
        return -0.1
