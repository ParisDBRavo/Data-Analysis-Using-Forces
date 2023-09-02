import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import Forces
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

def gettingLastDistances(LocationOfSites,lastPositionDataSet, forceToUse):
    cwd = Path.cwd()
    out_path_gen = Path(cwd, "Data/BellsOut/")
    out_path_gen.mkdir(parents=True, exist_ok=True)
    name=archiveName(forceToUse)
    print(name)
    out_path_files = Path(cwd, ("Data/BellsOut/"+name))
    print(out_path_files)
    file_save = out_path_files 
    for pair in itertools.combinations(lastPositionDataSet.index, 2):
        x1,y1=LocationOfSites[pair[0]]
        x2,y2=LocationOfSites[pair[1]]
        lastPositionDataSet[pair[0]][pair[1]]=calculateDistanceBetweenTwoPoints(LocationOfSites[pair[0]],LocationOfSites[pair[1]])
        lastPositionDataSet[pair[1]][pair[0]]=(lastPositionDataSet[pair[0]][pair[1]])
    #print(lastPositionDataSet)
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

def calculateForceMagnitud(pair, dataset_I, forceToUse):
    match forceToUse:
        case 1:
            #print("Using force where the length of types of bells are reduced based on which are encountered on the sites.")
            return Forces.forceReducingZeroesConstantRepelent(pair, dataset_I)
        case 2:
            #print("Using force where repelent with number of bells not equal and attraction with equal number.")
            return Forces.forceRepelentNonequal(pair, dataset_I)
        case 3:
            #print("Using force where repelent and attraction same number")
            return Forces.forceAtractionEqualRepelent(pair, dataset_I)
        case 4:
            #print("Using force where repelent and attraction same number of multiplication of masses.")
            return Forces.forceSameGravity(pair, dataset_I)
        case _:
            #print("Using default case of force constant repelent force and attraction of number of common bells.")
            return Forces.forceConstantRepelent(pair, dataset_I)

def archiveName(forceToUse):
    match forceToUse:
        case 1:
            print("Using force where the length of types of bells are reduced based on which are encountered on the sites.")
            return("ForceLengthReduced.csv")
        case 2:
            print("Using force where repelent with number of bells not equal and attraction with equal number.")
            return("ForceRepelentNotEqual.csv")
        case 3:
            print("Using force where repelent and attraction same number")
            return("ForceRepelentAttractionSame.csv")
        case 4:
            print("Using force where repelent and attraction same number of multiplication of masses.")
            return("ForceMasses.csv")
        case _:
            print("Using default case of force constant repelent force and attraction of number of common bells.")
            return("DefaultForces.csv")

