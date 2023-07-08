import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import Tools
import threading
import time
def processTimeStep(points, dataset_I):
    
    forces =np.zeros((len(dataset_I),2))
    for pair in itertools.combinations(dataset_I["SiteName"], 2):
        distance=Tools.calculateDistanceBetweenTwoPoints(points[pair[0]], points[pair[1]])
        direction=Tools.calculateDirectionBetweenTwoPoints(points[pair[0]], points[pair[1]])
        force =Tools.calculateForceMagnitud1(pair,dataset_I,realDistanceCosideration)
        firstRowNumber = dataset_I.loc[dataset_I["SiteName"]==pair[0]].index[0]
        secondRowNumber = dataset_I.loc[dataset_I["SiteName"]==pair[1]].index[0]
        if distance !=0:
            alfa1=(1/distance)*0.5*force*direction[0]*t**2
            beta1=(1/distance)*0.5*force*direction[1]*t**2
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



        

# Define a function that represents the worker thread
def workerThread(startTime, endTime):
    for t in np.arange(startTime, endTime, timeStep):
        lock.acquire()  # Acquire the lock to access shared resources
        processTimeStep(points, dataset_I)
        lock.release()  # Release the lock after accessing shared resources

#Number of time steps I am going to use
Nt = 3
#Parameter that sometimes helps
courant = 1
#maximum time
maxTime=100
timeStep =courant*maxTime/Nt

realDistanceCosideration= False
cwd = Path.cwd()
file_I = Path("Data/OutGeneralInfo/bell_Sbin.csv")
file_open = cwd / file_I

dataset_I = pd.read_csv(file_open)
points = Tools.createRandomPoints(dataset_I["SiteName"])
constX, constY = Tools.gettingConstants(points)
Tools.printingImagesWithNames(points)

# Create a lock to synchronize access to shared resources
lock = threading.Lock()

numThreads = 10
threadRange = maxTime / numThreads
threads = []
# Create and start the worker threads
for t in np.arange(0,maxTime, timeStep):
    for i in range(numThreads):
        startTime = i * threadRange
        endTime = (i + 1) * threadRange
        thread = threading.Thread(target=workerThread, args=(startTime, endTime))
        thread.start()
        threads.append(thread)

# Wait for all threads to finish
    for thread in threads:
        thread.join()

    Tools.printingImagesWithNames(points)