import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import Tools
#Number of time steps I am going to use
Nt = 1000
#Parameter that sometimes helps
courant = 1
#maximum time
Ntmax=100
dt =courant*Ntmax/Nt

cwd = Path.cwd()
file_I = Path("Data/OutGeneralInfo/bell_Sbin.csv")
file_open = cwd / file_I

dataset_I = pd.read_csv(file_open)
points = Tools.createRandomPoints(dataset_I["SiteName"])
print(points)

velocity =Tools.initializeVelocity(points)
print(velocity)
constX, constY = Tools.gettingConstants(points)
Tools.printingImagesWithNames(points)

forceToUse = 4
lastPositionDataSet= Tools.creatingNewDataset(dataset_I)
import time
start_time = time.time()
for t in np.arange(0,Ntmax, dt):
    forces =np.zeros((len(dataset_I),2))
    for pair in itertools.combinations(dataset_I["SiteName"], 2):
        distance=Tools.calculateDistanceBetweenTwoPoints(points[pair[0]], points[pair[1]])
        direction=Tools.calculateDirectionBetweenTwoPoints(points[pair[0]], points[pair[1]])
        force =Tools.calculateForceMagnitud(pair,dataset_I, forceToUse)
        if force == 4:
            massFirstSite, massSecondSite =Tools.getmasses(pair, dataset_I)
        else:
            massFirstSite =1
            massSecondSite =1
        firstRowNumber = dataset_I.loc[dataset_I["SiteName"]==pair[0]].index[0]
        secondRowNumber = dataset_I.loc[dataset_I["SiteName"]==pair[1]].index[0]
        if distance !=0:
            alfa1=((1/distance)*0.5*force*direction[0]*dt**2)/massSecondSite
            beta1=((1/distance)*0.5*force*direction[1]*dt**2)/massFirstSite
        else:
            alfa1=0.0
            beta1=0.0
        forces[firstRowNumber]+= np.asarray((alfa1,beta1))
        forces[secondRowNumber]+= np.asarray((-alfa1,-beta1))
        #if firstRowNumber==1:
        #    print("Forces=",forces[firstRowNumber])
        #    print("alfa=", alfa1)
        #    print("beta=", beta1)
    for i, siteNames in enumerate(dataset_I["SiteName"]):
        x1=constX[i]+forces[i][0]
        y1=constY[i]+forces[i][1]
        points[siteNames]=np.asarray((x1,y1))
    print(t)
    #Tools.printingImagesWithNames(points)
    #print(t)
end_time = time.time()
elapsed_time = end_time - start_time
print(elapsed_time)
Tools.printingImagesWithNames(points)
Tools.gettingLastDistances(points,lastPositionDataSet, forceToUse)
