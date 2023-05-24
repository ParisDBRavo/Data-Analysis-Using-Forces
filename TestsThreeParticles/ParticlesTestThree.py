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

def createNonrandomPoints(siteNames):
    points={}
    i=0
    j=0
    for item in siteNames:
        points[item] = np.array([i,j])
        if i==1:
            j=+1
            i=0
        i=+1
    return points
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

def solvingMotionEquations(points):
    
    return points

def calculateDistanceBetweenTwoPoints(firstPoint, secondPoint):
    print("Resta",firstPoint - secondPoint)
    print("distancia",np.linalg.norm(firstPoint - secondPoint))
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
        return -0.01
    #if number>=M/2:
    #    return number
    #else:
    #    return number-0.5
def calculateForceMagnitud1(pair, dataset_I):
    M=dataset_I.shape[1]-1
    firstRowNumber = dataset_I.loc[dataset_I["SiteName"]==pair[0]].index[0]
    secondRowNumber = dataset_I.loc[dataset_I["SiteName"]==pair[1]].index[0]
    dataset_I = dataset_I.drop(dataset_I.columns[0], axis = 1)
    number= np.count_nonzero((dataset_I.loc[firstRowNumber,:].astype(int))&(dataset_I.loc[secondRowNumber,:].astype(int)))/M
    if number!=0:
        return number
    else:
        return -0.1
#Number of time steps I am going to use
Nt = 10
#Parameter that sometimes helps
courant = 1
#maximum time
Ntmax=10
dt =courant*Ntmax/Nt

cwd = Path.cwd()
file_I = Path("Data/ThreePointsData/first_three_point_try.csv")
file_open = cwd / file_I

dataset_I = pd.read_csv(file_open)
points = createNonrandomPoints(dataset_I["SiteName"])
constX, constY = gettingConstants(points)
printingImagesWithNames(points)
print()
for t in np.arange(0,Ntmax, dt):
    forces =np.zeros((len(dataset_I),2))
    for pair in itertools.combinations(dataset_I["SiteName"], 2):
        distance=calculateDistanceBetweenTwoPoints(points[pair[0]], points[pair[1]])
        direction=calculateDirectionBetweenTwoPoints(points[pair[0]], points[pair[1]])
        #This force is the one between the arrays of the sites
        force =calculateForceMagnitud1(pair,dataset_I)
        #Calculate the i,j position of boths sites in the 
        print("Points = ",(pair))
        print("Distance = ", (distance))
        print("Direction = ", (direction))
        print("Force=",force)
        
        firstRowNumber = dataset_I.loc[dataset_I["SiteName"]==pair[0]].index[0]
        secondRowNumber = dataset_I.loc[dataset_I["SiteName"]==pair[1]].index[0]
        #This piece of the code should be different, it deals for divisions of 0
        if distance !=0:
            alfa1=(1/distance)*0.5*force*direction[0]*t**2
            beta1=(1/distance)*0.5*force*direction[1]*t**2
            print("entrooo")
        else:
            alfa1=0.0
            beta1=0.0
        #Adds boths terms in each site for later use the terms are the ones of 0.5*t**2
        forces[firstRowNumber]+= np.asarray((alfa1,beta1))
        forces[secondRowNumber]+= np.asarray((-alfa1,-beta1))
        #if firstRowNumber==1:
        #    print("Forces=",forces[firstRowNumber])
        print("alfa=", alfa1)
        print("beta=", beta1)
        print("***********************")
    #End part where each particle should have it forces of others
    for i, siteNames in enumerate(dataset_I["SiteName"]):
        x1=constX[i]+forces[i][0]
        y1=constY[i]+forces[i][1]
        
        print("x1",x1)
        print("y1",y1)
        print("***********************")
        points[siteNames]=np.asarray((x1,y1))


    
    printingImagesWithNames(points)
    #print(t)
