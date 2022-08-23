#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
GMMs and image classification
'''


# In[2]:


import numpy as np
import random as rnd
import scipy
from matplotlib import pyplot as plt
from sklearn.metrics import plot_confusion_matrix
import seaborn as sn
import pandas as pd
import math as m
import os
from sklearn import preprocessing
from scipy.stats import norm


# In[3]:


def bivariate_normal(X, Y, sigmax=1.0, sigmay=1.0,
                 mux=0.0, muy=0.0, sigmaxy=0.0): #3rd and 4th inputs are the std dev and the 7th input is the cov term only
    Xmu = X-mux
    Ymu = Y-muy

    rho = sigmaxy/(sigmax*sigmay)
    z = Xmu**2/sigmax**2 + Ymu**2/sigmay**2 - 2*rho*Xmu*Ymu/(sigmax*sigmay)
    denom = 2*np.pi*sigmax*sigmay*np.sqrt(1-rho**2)
    return np.exp(-z/(2*(1-rho**2))) / denom


# In[4]:


#K-means cloustering for the synthetic data as its easier to visualise
#CHANGE THE FILE NAMES BELOW

file=open(r"19\train.txt","r")
c=0
c1c=0
c2c=0
traindata_class1=np.zeros((1250,2))
traindata_class2=np.zeros((1250,2))

for line in file:
    line=(list(map(float, line.strip().split(","))))
    c+=1
    if int(line[2])==1:
        traindata_class1[c1c]=line[:2]
        c1c+=1
    else:
        traindata_class2[c2c]=line[:2]
        c2c+=1
        
K=10 #K means with 10 clusters


# In[5]:


#CHANGE THE FILE NAMES BELOW
file=open(r"19\dev.txt","r")
c=0
c1c=0
c2c=0
devdata_class1=np.zeros((500,2))
devdata_class2=np.zeros((500,2))

for line in file:
    line=(list(map(float, line.strip().split(","))))
    c+=1
    if int(line[2])==1:
        devdata_class1[c1c]=line[:2]
        c1c+=1
    else:
        devdata_class2[c2c]=line[:2]
        c2c+=1


# In[6]:


def maxminofdimension(traindata_class1):
    maxminofdimension_=np.zeros((len(traindata_class1[0]),2))
    for i in range(len(maxminofdimension_)):
            maxminofdimension_[i][0]=max(traindata_class1[:,i])
            maxminofdimension_[i][1]=min(traindata_class1[:,i])
    return maxminofdimension_


# In[7]:


def Kmeans(K, traindata_class1):
    
    #initialising the variables
    ndim=len(traindata_class1[0])
    n=len(traindata_class1)
    gammas=np.zeros((n,K))
    l=0
    pointsincluster=[ [] for l in range(K) ]
    mmofd=maxminofdimension(traindata_class1)
    means=np.zeros((K,ndim))
    means1=np.zeros((K,ndim))
    numofpoints=[0]*K
    
    #initialising the centers of the clusters
    for i in range(K):
        #for k in range(ndim):
            #mmofd[k][1]+rnd.random()*(mmofd[k][0]-mmofd[k][1])
        means[i]=traindata_class1[int(rnd.random()*len(traindata_class1))]
    #EM step
    itercounter=0
    while(itercounter<10):
        
        #initialisation
        l=0
        pointsincluster=[ [] for l in range(K) ]
        numofpoints=[0]*K
        gammas=np.zeros((n,K))
        c=0
        
        #finding which point goes to which class
        for i in traindata_class1:
            dist=i-means
            sumofsquares=np.zeros((len(dist),1))
            for k in range(ndim):
                sumofsquares+=np.c_[dist[:,k]*dist[:,k]]
            eucdist=np.sqrt(sumofsquares)
            ibelongstocluster= np.argmin(eucdist)
            numofpoints[ibelongstocluster]+=1
            gammas[c][ibelongstocluster]+=1
            pointsincluster[ibelongstocluster].append(list(i))
            c+=1
        pointsincluster1=np.array(pointsincluster)
        
        
        '''
        xt1, yt1 = np.array(pointsincluster1[0]).T
        plt.scatter(xt1,yt1)
        x1, y1 = means.T
        plt.scatter(x1,y1)   
        plt.show()
        '''

        #updating the means
        for i in range(K):
            for k in range(ndim):
                if len(np.array(pointsincluster1[i]).shape)!=1:
                    means[i][k]=np.sum(np.array(pointsincluster1[i])[:,k])/numofpoints[i]
        itercounter+=1
        
    return([means,pointsincluster,gammas])


# In[8]:


nclasses=2
K=20
ndim=2

meansclass1,pointstemp1,gammatemp1=Kmeans(K, traindata_class1)
meansclass2,pointstemp2,gammatemp2=Kmeans(K, traindata_class2)
meansall=np.zeros((nclasses,K,ndim))
meansall=meansclass1,meansclass2


# In[9]:


xt1, yt1 = traindata_class1.T
plt.scatter(xt1,yt1)
x1, y1 = meansclass1.T
plt.scatter(x1,y1)

xt2, yt2 = traindata_class2.T
plt.scatter(xt2,yt2)
x2, y2 = meansclass2.T
plt.scatter(x2,y2)
'''
x2, y2 = meansclass2.T
plt.scatter(x2,y2)
xt2, yt2 = traindata_class2.T
plt.scatter(xt2,yt2)
'''
plt.show()


# In[10]:


#plotting confusion matrix
cm2=np.zeros((2,2))

#finding the accuracy for class1

correct=0
false=0
for i in devdata_class1:
    mineucdist=1e9
    category=0
    for p in range(len(meansall)):
        dist=i-meansall[p]
        sumofsquares=np.zeros((len(dist),1))
        for k in range(ndim):
            sumofsquares+=np.c_[dist[:,k]*dist[:,k]]
        eucdist=np.sqrt(sumofsquares)
        if(np.amin(eucdist) < mineucdist):
            mineucdist=np.amin(eucdist)
            category=p+1   
    if category==1:
        correct+=1
    else:
        false+=1
cm2[0][0]=correct
cm2[0][1]=false


# In[11]:


#finding the accuracy for class2

correct=0
false=0
for i in devdata_class2:
    mineucdist=1e9
    category=0
    for p in range(len(meansall)):
        dist=i-meansall[p]
        sumofsquares=np.zeros((len(dist),1))
        for k in range(ndim):
            sumofsquares+=np.c_[dist[:,k]*dist[:,k]]
        eucdist=np.sqrt(sumofsquares)
        if(np.amin(eucdist) < mineucdist):
            mineucdist=np.amin(eucdist)
            category=p+1   
    if category==2:
        correct+=1
    else:
        false+=1
cm2[1][0]=false
cm2[1][1]=correct


# In[12]:


#plot of dev data vs the clusters centers obtained

xt1, yt1 = devdata_class1.T
plt.scatter(xt1,yt1,color="purple")
x1, y1 = meansclass1.T
plt.scatter(x1,y1,color="orange")

xt2, yt2 = devdata_class2.T
plt.scatter(xt2,yt2,color="cyan")
x2, y2 = meansclass2.T
plt.scatter(x2,y2,color="red")
'''
x2, y2 = meansclass2.T
plt.scatter(x2,y2)
xt2, yt2 = traindata_class2.T
plt.scatter(xt2,yt2)
'''
plt.show()


# In[24]:


#confusion matrix

df_cm = pd.DataFrame(cm2, index = [i for i in ["predicted as class1","predicted as class2"]],
                  columns = [i for i in ["actual class1","actual class2"]])
plt.figure(figsize = (2,2))
sn.heatmap(df_cm, annot=True, fmt="g", cmap="cool")
plt.show()


# In[25]:


#errors vs K

nclasses=2
ndim=2
accvsk=[]

for K in range(2,30):
    meansclass1,pointstemp1,gammatemp1=Kmeans(K, traindata_class1)
    meansclass2,pointstemp2,gammatemp2=Kmeans(K, traindata_class2)
    meansall=np.zeros((nclasses,K,ndim))
    meansall=meansclass1,meansclass2
    #plotting confusion matrix
    
    cm2=np.zeros((2,2))

    #finding the accuracy for class1

    correct=0
    false=0
    for i in devdata_class1:
        mineucdist=1e9
        category=0
        for p in range(len(meansall)):
            dist=i-meansall[p]
            sumofsquares=np.zeros((len(dist),1))
            for k in range(ndim):
                sumofsquares+=np.c_[dist[:,k]*dist[:,k]]
            eucdist=np.sqrt(sumofsquares)
            if(np.amin(eucdist) < mineucdist):
                mineucdist=np.amin(eucdist)
                category=p+1   
        if category==1:
            correct+=1
        else:
            false+=1
    cm2[0][0]=correct
    cm2[0][1]=false
    
    #finding the accuracy for class2

    correct=0
    false=0
    for i in devdata_class2:
        mineucdist=1e9
        category=0
        for p in range(len(meansall)):
            dist=i-meansall[p]
            sumofsquares=np.zeros((len(dist),1))
            for k in range(ndim):
                sumofsquares+=np.c_[dist[:,k]*dist[:,k]]
            eucdist=np.sqrt(sumofsquares)
            if(np.amin(eucdist) < mineucdist):
                mineucdist=np.amin(eucdist)
                category=p+1   
        if category==2:
            correct+=1
        else:
            false+=1
    cm2[1][0]=false
    cm2[1][1]=correct
    
    accuracy=(cm2[0][0]+cm2[1][1])/(cm2[1][0]+cm2[0][1]+cm2[0][0]+cm2[1][1])
    accvsk.append([K,accuracy])

    #confusion matrix
    
    if K in [3,5,7,9,10,15,20,28]:
        df_cm = pd.DataFrame(cm2, index = [i for i in ["predicted as class1","predicted as class2"]],
                          columns = [i for i in ["actual class1","actual class2"]])
        plt.figure(figsize = (2,2))
        plt.title("K="+str(K))
        sn.heatmap(df_cm, annot=True, fmt="g", cmap="cool")

        print("accuracy for K="+str(K)+" is "+str(accuracy))


# In[15]:

plt.show()
plt.figure()
accvsk=np.array(accvsk)
xaccvsk, yaccvsk = accvsk.T
plt.scatter(xaccvsk,yaccvsk,color="black")
plt.plot(accvsk[:,0],accvsk[:,1],color="purple")
plt.grid()
plt.xticks(np.arange(min(accvsk[:,0]), max(accvsk[:,0])+1, 1.0))
plt.title("accuracy vs K for K means")
plt.show()


# In[16]:


#We can see that K>=10 given us the best accuracy of approx 100%


# In[17]:


#function that returns the means and covariance matrix given the points and weights
def covariance(arr,weights):
    ndim = len(arr[0])
    n=len(arr)
    means = np.zeros(ndim)
    
    #finding the means
    for i in range(ndim):
        s=0
        sweights=0
        for k in range(n):
            s+=arr[k][i]*weights[k]
            sweights+=weights[k]
        means[i]= (s/sweights)
    
    #finding the covariances
    cov = np.zeros((ndim,ndim))
    
    arrminusmean=arr-means
    for i in range(ndim):
        for k in range(ndim):
            cov[i][k]=np.sum(weights*arrminusmean[:,i]*arrminusmean[:,k])/sweights
    return[means,cov]


# In[18]:


#Now we proceed with GMMs

def GMMs(K, traindata_class1, givenmeans, givenpointsincluster, givengammas):
    
    n = len(traindata_class1)
    ndim = len(traindata_class1[0])
    
    #initialising the variables (we use the points and points that we got via kmeans), this reduces the number of iterations
    meansofKclusters = givenmeans.copy() #np.zeros((K,ndim))
    CovarianceMatrixofclusters=np.zeros((K,ndim,ndim))
    gammas=givengammas.copy()
    pointsincluster= givenpointsincluster.copy() #[ [] for l in range(K) ]
    numofpoints=np.zeros(K)
    pies=np.zeros(K) #the 3.14 kind of pi
    
    for i in range(K):
        numofpoints[i]=len(givenpointsincluster[i])
        CovarianceMatrixofclusters[i]=covariance(traindata_class1,gammas[:,i])[1]
    for i in range(K):
        pies[i]= numofpoints[i]/n
    
    #EM step
    itercounter=0
    while(itercounter<5):
        
        #updating gammas
        
        for i in range(n):
            for k in range(K):
                gammas[i][k]= pies[k]*((-1* ( (2*np.pi)**(-ndim/2) )* np.exp(-0.5* ((traindata_class1[i]-meansofKclusters[k])@np.linalg.inv(CovarianceMatrixofclusters[k])@(traindata_class1[i]-meansofKclusters[k]).T)))/(np.linalg.det(CovarianceMatrixofclusters[k])))
        
        #normalising the values
        for i in range(n):
            gammas[i]=gammas[i]/np.sum(gammas[i])
            
        for k in range(K):
            pies[k]=np.sum(gammas[:,k])/n
            
        #updating the cluster centers and the covariance matrices and updating the pies
        for k in range(K):
            meansofKclusters[k],CovarianceMatrixofclusters[k]=covariance(traindata_class1,gammas[:,k])
        itercounter+=1
        
        '''
        plt.figure()
        xt1, yt1 = np.array(traindata_class1).T
        plt.scatter(xt1,yt1,color="yellow")
        x1, y1 = meansofKclusters.T
        plt.scatter(x1,y1,color="black") 
        
        for k in range(K):
            x = np.linspace(meansofKclusters[k][0]-1, meansofKclusters[k][0]+1, 100)
            y = np.linspace(meansofKclusters[k][1]-1, meansofKclusters[k][1]+1, 100)
            XX, YY = np.meshgrid(x, y)
            Z = bivariate_normal(XX, YY, np.sqrt(CovarianceMatrixofclusters[k][0][0]), np.sqrt(CovarianceMatrixofclusters[k][1][1]), meansofKclusters[k][0], meansofKclusters[k][1], CovarianceMatrixofclusters[k][1][0])
            plt.contour(XX, YY, Z)
        plt.show()
        '''
        return([meansofKclusters,CovarianceMatrixofclusters,gammas])


# In[19]:


#making diagonal covariance matrices
def covdiag(cov):
    for i in range(len(cov)):
        for j in range(len(cov[0])):
            if i!=j:
                cov[i][j]=0
    return(cov)


# In[20]:


#Caluclating GMMs when the covariance is diagonal

def GMMscovdiag(K, traindata_class1, givenmeans, givenpointsincluster, givengammas):
    
    n = len(traindata_class1)
    ndim = len(traindata_class1[0])
    
    #initialising the variables (we use the points and points that we got via kmeans), this reduces the number of iterations
    meansofKclusters = givenmeans.copy() #np.zeros((K,ndim))
    CovarianceMatrixofclusters=np.zeros((K,ndim,ndim))
    gammas=givengammas.copy()
    pointsincluster= givenpointsincluster.copy() #[ [] for l in range(K) ]
    numofpoints=np.zeros(K)
    pies=np.zeros(K) #the 3.14 kind of pi
    
    for i in range(K):
        numofpoints[i]=len(givenpointsincluster[i])
        CovarianceMatrixofclusters[i]=covdiag(covariance(traindata_class1,gammas[:,i])[1])
    for i in range(K):
        pies[i]= numofpoints[i]/n
    
    #EM step
    itercounter=0
    while(itercounter<5):
        
        #updating gammas
        
        for i in range(n):
            for k in range(K):
                gammas[i][k]= pies[k]*((-1* ( (2*np.pi)**(-ndim/2) )* np.exp(-0.5* ((traindata_class1[i]-meansofKclusters[k])@np.linalg.inv(CovarianceMatrixofclusters[k])@(traindata_class1[i]-meansofKclusters[k]).T)))/(np.linalg.det(CovarianceMatrixofclusters[k])))
        
        #normalising the values
        for i in range(n):
            gammas[i]=gammas[i]/np.sum(gammas[i])
            
        for k in range(K):
            pies[k]=np.sum(gammas[:,k])/n
            
        #updating the cluster centers and the covariance matrices and updating the pies
        for k in range(K):
            meansofKclusters[k]=covariance(traindata_class1,gammas[:,k])[0]
            CovarianceMatrixofclusters[k]=covdiag(covariance(traindata_class1,gammas[:,k])[1])
        itercounter+=1
        
        return([meansofKclusters,CovarianceMatrixofclusters,gammas])


# In[21]:


GMMmeanclass1,GMMcovclass1,GMMgammaclass1=GMMs(K, traindata_class1, meansclass1, pointstemp1, gammatemp1)


# In[22]:


GMMmeanclass2,GMMcovclass2,GMMgammaclass2=GMMs(K, traindata_class2, meansclass2, pointstemp2, gammatemp2)


# In[27]:


#decision boundary

X=np.linspace(-16,4,500)
Y=np.linspace(-16,4,500)
boundary=[]
plt.figure()
for i in X:
    for j in Y:
        point=np.array([i,j])
        sumofsquares1=np.zeros((K,1))
        sumofsquares2=np.zeros((K,1))
        distpointc1=point-GMMmeanclass1
        distpointc2=point-GMMmeanclass2
        for k in range(ndim):
            sumofsquares1+=np.c_[distpointc1[:,k]*distpointc1[:,k]]
            sumofsquares2+=np.c_[distpointc2[:,k]*distpointc2[:,k]]
        eucdistc1=np.sqrt(sumofsquares1)
        eucdistc2=np.sqrt(sumofsquares2)
        if(abs(np.amin(eucdistc1) - np.amin(eucdistc2))<0.05):
            boundary.append(list(point))
xd, yd = np.array(boundary).T
plt.scatter(xd,yd,color="black") 
plt.show()


# In[ ]:


#final plot of data and K gmm obtained gaussians
plt.figure()
xt1, yt1 = np.array(traindata_class1).T
plt.scatter(xt1,yt1,color="yellow")
x1, y1 = GMMmeanclass1.T
plt.scatter(x1,y1,color="black") 

xt2, yt2 = np.array(traindata_class2).T
plt.scatter(xt2,yt2,color="red")
x2, y2 = GMMmeanclass2.T
plt.scatter(x2,y2,color="black")

for k in range(K):
    x = np.linspace(GMMmeanclass1[k][0]-1, GMMmeanclass1[k][0]+1, 100)
    y = np.linspace(GMMmeanclass1[k][1]-1, GMMmeanclass1[k][1]+1, 100)
    XX, YY = np.meshgrid(x, y)
    Z = bivariate_normal(XX, YY, np.sqrt(GMMcovclass1[k][0][0]), np.sqrt(GMMcovclass1[k][1][1]), GMMmeanclass1[k][0], GMMmeanclass1[k][1], (GMMcovclass1[k][1][0]))
    plt.contour(XX, YY, Z)
    
for k in range(K):
    x = np.linspace(GMMmeanclass2[k][0]-1, GMMmeanclass2[k][0]+1, 100)
    y = np.linspace(GMMmeanclass2[k][1]-1, GMMmeanclass2[k][1]+1, 100)
    XX, YY = np.meshgrid(x, y)
    Z = bivariate_normal(XX, YY, np.sqrt(GMMcovclass2[k][0][0]), np.sqrt(GMMcovclass2[k][1][1]), GMMmeanclass2[k][0], GMMmeanclass2[k][1], (GMMcovclass2[k][1][0]))
    plt.contour(XX, YY, Z)
    
plt.title("Traindata+ gmm contours")
plt.show()


# In[ ]:


#final plot of data and K gmm obtained gaussians and decision boundary
plt.figure()
xt1, yt1 = np.array(traindata_class1).T
plt.scatter(xt1,yt1,color="yellow")
x1, y1 = GMMmeanclass1.T
plt.scatter(x1,y1,color="black") 

xt2, yt2 = np.array(traindata_class2).T
plt.scatter(xt2,yt2,color="red")
x2, y2 = GMMmeanclass2.T
plt.scatter(x2,y2,color="black")

for k in range(K):
    x = np.linspace(GMMmeanclass1[k][0]-1, GMMmeanclass1[k][0]+1, 100)
    y = np.linspace(GMMmeanclass1[k][1]-1, GMMmeanclass1[k][1]+1, 100)
    XX, YY = np.meshgrid(x, y)
    Z = bivariate_normal(XX, YY, np.sqrt(GMMcovclass1[k][0][0]), np.sqrt(GMMcovclass1[k][1][1]), GMMmeanclass1[k][0], GMMmeanclass1[k][1], (GMMcovclass1[k][1][0]))
    plt.contour(XX, YY, Z)
    
for k in range(K):
    x = np.linspace(GMMmeanclass2[k][0]-1, GMMmeanclass2[k][0]+1, 100)
    y = np.linspace(GMMmeanclass2[k][1]-1, GMMmeanclass2[k][1]+1, 100)
    XX, YY = np.meshgrid(x, y)
    Z = bivariate_normal(XX, YY, np.sqrt(GMMcovclass2[k][0][0]), np.sqrt(GMMcovclass2[k][1][1]), GMMmeanclass2[k][0], GMMmeanclass2[k][1], (GMMcovclass2[k][1][0]))
    plt.contour(XX, YY, Z)
    
xd, yd = np.array(boundary).T
plt.scatter(xd,yd,color="black",s=5) 
plt.title("Traindata+decsion boundary+gmm contours")
    
plt.show()


# In[ ]:


#final plot of data and K gmm obtained gaussians and decision boundary
plt.figure()
xt1, yt1 = np.array(devdata_class1).T
plt.scatter(xt1,yt1,color="cyan")
x1, y1 = GMMmeanclass1.T
plt.scatter(x1,y1,color="black") 

xt2, yt2 = np.array(devdata_class2).T
plt.scatter(xt2,yt2,color="purple")
x2, y2 = GMMmeanclass2.T
plt.scatter(x2,y2,color="black")

for k in range(K):
    x = np.linspace(GMMmeanclass1[k][0]-1, GMMmeanclass1[k][0]+1, 100)
    y = np.linspace(GMMmeanclass1[k][1]-1, GMMmeanclass1[k][1]+1, 100)
    XX, YY = np.meshgrid(x, y)
    Z = bivariate_normal(XX, YY, np.sqrt(GMMcovclass1[k][0][0]), np.sqrt(GMMcovclass1[k][1][1]), GMMmeanclass1[k][0], GMMmeanclass1[k][1], (GMMcovclass1[k][1][0]))
    plt.contour(XX, YY, Z)
    
for k in range(K):
    x = np.linspace(GMMmeanclass2[k][0]-1, GMMmeanclass2[k][0]+1, 100)
    y = np.linspace(GMMmeanclass2[k][1]-1, GMMmeanclass2[k][1]+1, 100)
    XX, YY = np.meshgrid(x, y)
    Z = bivariate_normal(XX, YY, np.sqrt(GMMcovclass2[k][0][0]), np.sqrt(GMMcovclass2[k][1][1]), GMMmeanclass2[k][0], GMMmeanclass2[k][1], (GMMcovclass2[k][1][0]))
    plt.contour(XX, YY, Z)
    
xd, yd = np.array(boundary).T
plt.scatter(xd,yd,color="black",s=5) 
plt.title("Dev data+decsion boundary+gmm contours")
    
plt.show()


# In[ ]:


# classifying the points:

def classifiersyn(K,datapoint,alltrainmeans,alltraincov,allpies):
    c=0
    numblocks=36
    temp3=-1
    temp1=0
    prob2classes=np.zeros(2)
    for c in range(2):
        temp2=0
        temp2 += np.log(np.sum([allpies[c][j]*np.exp(-0.5*((datapoint-alltrainmeans[c][j]).T@np.linalg.pinv(alltraincov[c][j])@(datapoint-alltrainmeans[c][j])))/(np.linalg.det(alltraincov[c][j]))**0.5 for j in range(K)]))
        #print(temp2)
        if temp2>temp1:
            temp1 = temp2 
            temp3 = c
        prob2classes[c]=temp2
    prob2classes=prob2classes/np.sum(prob2classes)
    return([temp3,prob2classes])

def pointsclassifierloadersyn(K,devdata,cm,j,alltrainmeans,alltraincov,allpies):
    temp=[0]*len(devdata)
    for i in range(len(devdata)):
        temp[i]=classifiersyn(K, devdata[i],alltrainmeans,alltraincov,allpies)[0]
    for i in temp:
        cm[j][i]+=1
    print(cm[j])


# In[ ]:


accvskgmm=[]

for K in range(2,10):
    meansclass1,pointstemp1,gammatemp1=Kmeans(K, traindata_class1)
    meansclass2,pointstemp2,gammatemp2=Kmeans(K, traindata_class2)

    GMMmeanclass1,GMMcovclass1,GMMgammaclass1=GMMs(K, traindata_class1, meansclass1, pointstemp1, gammatemp1)
    GMMmeanclass2,GMMcovclass2,GMMgammaclass2=GMMs(K, traindata_class2, meansclass2, pointstemp2, gammatemp2)

    cmsynthetic=np.zeros((2,2))
    alltrainmeanssyn=np.zeros((2,K,2))
    alltraincovsyn=np.zeros((2,K,2,2))
    allpiessyn=np.zeros((2,K))

    for k in range(K):
        allpiessyn[0][k]=np.sum(GMMgammaclass1[:,k])
        allpiessyn[1][k]=np.sum(GMMgammaclass2[:,k])
    for i in range(2):
        allpiessyn[i]/(np.sum(allpiessyn[i]))

    #means
    alltrainmeanssyn[0]=GMMmeanclass1
    alltrainmeanssyn[1]=GMMmeanclass2

    #covariances
    alltraincovsyn[0]=GMMcovclass1
    alltraincovsyn[1]=GMMcovclass2


    #GMMmeanclass1,GMMcovclass1,GMMgammaclass1
    pointsclassifierloadersyn(K,devdata_class1,cmsynthetic,0,alltrainmeanssyn,alltraincovsyn,allpiessyn)
    pointsclassifierloadersyn(K,devdata_class2,cmsynthetic,1,alltrainmeanssyn,alltraincovsyn,allpiessyn)
    
    #confusion matrix
    
    accuracy=(cmsynthetic[0][0]+cmsynthetic[1][1])/(cmsynthetic[1][0]+cmsynthetic[0][1]+cmsynthetic[0][0]+cmsynthetic[1][1])
    accvskgmm.append([K,accuracy])

    if K in [3,5,7,9]:
        df_cm = pd.DataFrame(cmsynthetic, index = [i for i in ["predicted as class1","predicted as class2"]],
                          columns = [i for i in ["actual class1","actual class2"]])
        plt.figure(figsize = (2,2))
        plt.title("K="+str(K))
        sn.heatmap(df_cm, annot=True, fmt="g", cmap="cool")
        plt.show()

        print("accuracy for K="+str(K)+" is "+str(accuracy))


# In[ ]:

plt.figure()
accvskgmm=np.array(accvskgmm)
xaccvskgmm, yaccvskgmm = accvskgmm.T
plt.scatter(xaccvskgmm,yaccvskgmm,color="black")
plt.plot(accvskgmm[:,0],accvskgmm[:,1],color="purple")
plt.grid()
plt.xticks(np.arange(min(accvskgmm[:,0]), max(accvskgmm[:,0])+1, 1.0))
plt.title("accuracy vs K for GMMs")
plt.show()


# In[ ]:


#We can see that K>=5 given us the best accuracy of approx 100% in GMMs whereas it was K=>10 in K means
#5 clusters makes sense aswell if we visualise the data


# In[ ]:


#decision boundary of gmm

# '''
# X=np.linspace(-16,4,500)
# Y=np.linspace(-16,4,500)
# boundary=[]
# for i in X:
#     for j in Y:
#         point=np.array([i,j])
#         sumofsquares1=np.zeros((20,1))
#         sumofsquares2=np.zeros((20,1))
#         distpointc1=point-GMMmeanclass1
#         distpointc2=point-GMMmeanclass2
#         for k in range(ndim):
#             sumofsquares1+=np.c_[distpointc1[:,k]*distpointc1[:,k]]
#             sumofsquares2+=np.c_[distpointc2[:,k]*distpointc2[:,k]]
#         eucdistc1=np.sqrt(sumofsquares1)
#         eucdistc2=np.sqrt(sumofsquares2)
#         if(abs(np.amin(eucdistc1) - np.amin(eucdistc2))<0.05):
#             boundary.append(list(point))
# xd, yd = np.array(boundary).T
# plt.scatter(xd,yd,color="black") 
# '''


# In[ ]:


K=20

meansclass1,pointstemp1,gammatemp1=Kmeans(K, traindata_class1)
meansclass2,pointstemp2,gammatemp2=Kmeans(K, traindata_class2)

GMMmeanclass1,GMMcovclass1,GMMgammaclass1=GMMs(K, traindata_class1, meansclass1, pointstemp1, gammatemp1)
GMMmeanclass2,GMMcovclass2,GMMgammaclass2=GMMs(K, traindata_class2, meansclass2, pointstemp2, gammatemp2)

cmsynthetic=np.zeros((2,2))
alltrainmeanssyn=np.zeros((2,K,2))
alltraincovsyn=np.zeros((2,K,2,2))
allpiessyn=np.zeros((2,K))

for k in range(K):
    allpiessyn[0][k]=np.sum(GMMgammaclass1[:,k])
    allpiessyn[1][k]=np.sum(GMMgammaclass2[:,k])
for i in range(2):
    allpiessyn[i]/(np.sum(allpiessyn[i]))

#means
alltrainmeanssyn[0]=GMMmeanclass1
alltrainmeanssyn[1]=GMMmeanclass2

#covariances
alltraincovsyn[0]=GMMcovclass1
alltraincovsyn[1]=GMMcovclass2


#GMMmeanclass1,GMMcovclass1,GMMgammaclass1
pointsclassifierloadersyn(K,devdata_class1,cmsynthetic,0,alltrainmeanssyn,alltraincovsyn,allpiessyn)
pointsclassifierloadersyn(K,devdata_class2,cmsynthetic,1,alltrainmeanssyn,alltraincovsyn,allpiessyn)


# In[ ]:


# #final plot of data and K gmm obtained gaussians and decision boundary
# plt.figure()
# xt1, yt1 = np.array(devdata_class1).T
# plt.scatter(xt1,yt1,color="cyan")
# x1, y1 = GMMmeanclass1.T
# plt.scatter(x1,y1,color="black") 

# xt2, yt2 = np.array(devdata_class2).T
# plt.scatter(xt2,yt2,color="purple")
# x2, y2 = GMMmeanclass2.T
# plt.scatter(x2,y2,color="black")

# for k in range(K):
#     x = np.linspace(GMMmeanclass1[k][0]-1, GMMmeanclass1[k][0]+1, 100)
#     y = np.linspace(GMMmeanclass1[k][1]-1, GMMmeanclass1[k][1]+1, 100)
#     XX, YY = np.meshgrid(x, y)
#     Z = bivariate_normal(XX, YY, np.sqrt(GMMcovclass1[k][0][0]), np.sqrt(GMMcovclass1[k][1][1]), GMMmeanclass1[k][0], GMMmeanclass1[k][1], (GMMcovclass1[k][1][0]))
#     plt.contour(XX, YY, Z)
    
# for k in range(K):
#     x = np.linspace(GMMmeanclass2[k][0]-1, GMMmeanclass2[k][0]+1, 100)
#     y = np.linspace(GMMmeanclass2[k][1]-1, GMMmeanclass2[k][1]+1, 100)
#     XX, YY = np.meshgrid(x, y)
#     Z = bivariate_normal(XX, YY, np.sqrt(GMMcovclass2[k][0][0]), np.sqrt(GMMcovclass2[k][1][1]), GMMmeanclass2[k][0], GMMmeanclass2[k][1], (GMMcovclass2[k][1][0]))
#     plt.contour(XX, YY, Z)
    
# xd, yd = np.array(boundary).T
# plt.scatter(xd,yd,color="black",s=5) 
# plt.title("Dev data+decsion boundary+gmm contours")
    
# plt.show()


# In[ ]:


def classifiersynexp(K,datapoint,alltrainmeans,alltraincov,allpies):
    c=0
    numblocks=36
    temp3=-1
    temp1=0
    prob2classes=[0]*2
    for c in range(2):
        temp2=0
        temp2 += (np.sum([allpies[c][j]*np.exp(-0.5*((datapoint-alltrainmeans[c][j]).T@np.linalg.pinv(alltraincov[c][j])@(datapoint-alltrainmeans[c][j])))/(np.linalg.det(alltraincov[c][j]))**0.5 for j in range(K)]))
        #print(temp2)
        if temp2>temp1:
            temp1 = temp2 
            temp3 = c
        prob2classes[c]=temp2
    prob2classes=prob2classes/np.sum(prob2classes)
    return([temp3,prob2classes])


# In[ ]:


#getting the scores for the ROC

scores=[]
for i in devdata_class1:
    temp1,temp2=classifiersynexp(K,i,alltrainmeanssyn,alltraincovsyn,allpiessyn)
    scores.append((temp2[0],0,temp1))
    scores.append((temp2[1],1,temp1))
for i in devdata_class2:
    temp1,temp2=classifiersynexp(K,i,alltrainmeanssyn,alltraincovsyn,allpiessyn)
    scores.append((temp2[0],0,temp1))
    scores.append((temp2[1],1,temp1))


# In[ ]:


#plotting the ROC

def plotROC(scores): 
    fig,axis = plt.subplots(1,figsize=(10,8))
    temp_list = []
    for score in scores : 
        temp_list.append(score[0])
    scores = sorted(scores)
    temp_list = sorted(temp_list)
    TPR = []  # TP/(TP+FN)
    FNR = []  # FN/(FN+TP)
    FPR = []  # FP/(FP+TN)

    for threshold in temp_list : 
        tp = 0
        tn = 0
        fp = 0
        fn = 0 
        for data in scores : 
            if(data[0] >= threshold) : 
                if(data[1] == data[2]) : 
                    tp += 1
                else : 
                    fp += 1
            else : 
                if(data[1] == data[2]) : 
                    fn += 1
                else :
                    tn += 1
        TPR.append(float(tp/(tp+fn)))
        FNR.append(float(fn/(fn+tp)))
        FPR.append(float(fp/(fp+tn)))
        
#     display=sklearn.metrics.DetCurveDisplay(FPR, FNR, estimator_name=None, pos_label=None)
#     display.plot()
#     plt.show()
            
    axis.plot(FPR,TPR)
    axis.set_title(f"ROC Curve")
    axis.set_xlabel("False Positive Rate(FPR)")
    axis.set_ylabel("True Positive Rate(TPR)")
    x=np.linspace(0,1,100)
    y=np.linspace(0,1,100)
    plt.plot(x,y)
    plt.plot()


# In[ ]:


def normalise(values):
    fmin = np.min(values,axis=0)
    fmax = np.max(values,axis=0)
    temp = (values - fmin)/(fmax-fmin)
    return temp


# In[ ]:


#plotting the DETs

def plotDET(scores): 
    fig,axis = plt.subplots(1,figsize=(10,8))
    temp_list = []
    for score in scores : 
        temp_list.append(score[0])
    scores = sorted(scores)
    temp_list = sorted(temp_list)
    TPR = []  # TP/(TP+FN)
    FNR = []  # FN/(FN+TP)
    FPR = []  # FP/(FP+TN)

    for threshold in temp_list : 
        tp = 0
        tn = 0
        fp = 0
        fn = 0 
        for data in scores : 
            if(data[0] >= threshold) : 
                if(data[1] == data[2]) : 
                    tp += 1
                else : 
                    fp += 1
            else : 
                if(data[1] == data[2]) : 
                    fn += 1
                else :
                    tn += 1
        TPR.append(float(tp/(tp+fn)))
        FNR.append(float(fn/(fn+tp)))
        FPR.append(float(fp/(fp+tn)))

    blah1 = norm.ppf(FPR)
    blah2 = norm.ppf(FNR)
    axis.plot(blah1,blah2)
    axis.set_title(f"DET Curve")
    axis.set_xlabel("False Alarm Rate")
    axis.set_ylabel("Missed Detection Rate")

    values = axis.get_yticks()
    axis.set_yticklabels(["{:.0%}".format(y) for y in normalise(values)])
    values = axis.get_xticks()
    axis.set_xticklabels(["{:.0%}".format(x) for x in normalise(values)])

    fig.savefig("DET_curve.png")


# In[ ]:


plotROC(scores)
plotDET(scores)
#scores


# In[ ]:


# Now finding the GMMs for the images


# In[ ]:


#function that loads the data
def dataloader(folder_path,loadeddata):
    #folder_path = r'PRML_Assignment3_data\coast\dev'
    i=0
    j=0
    k=0
    for filename in os.listdir(folder_path):
        with open(os.path.join(folder_path, filename), 'r') as f:
            text = f.read()
            temp=(list(map(str, text.strip().split("\n"))))
            for line in temp:
                temp1=(list(map(str, line.strip().split(","))))
                for line2 in temp1:
                    temp2=(list(map(float, line2.strip().split(" "))))
                    loadeddata[int(i/36),k%36,:]=temp2[:]
                    i+=1
                    k+=1
    return(loadeddata)


# In[ ]:


#loading all the data
#folder_path = r'PRML_Assignment3_data\coast\dev'
#coast forest highway mountain opencountry
#CHANGE THE FOLDER PATH NAMES BELOW ACCORDINGLY

#dev data
imgcoastdev=np.zeros((73,36,23))
imgcoastdev=dataloader(r'PRML_Assignment3_data\coast\dev',imgcoastdev)

imgforestdev=np.zeros((66,36,23))
imgforestdev=dataloader(r'PRML_Assignment3_data\forest\dev',imgforestdev)

imghighwaydev=np.zeros((52,36,23))
imghighwaydev=dataloader(r'PRML_Assignment3_data\highway\dev',imghighwaydev)

imgmountaindev=np.zeros((75,36,23))
imgmountaindev=dataloader(r'PRML_Assignment3_data\mountain\dev',imgmountaindev)

imgopencountrydev=np.zeros((82,36,23))
imgopencountrydev=dataloader(r'PRML_Assignment3_data\opencountry\dev',imgopencountrydev)

#train data
imgcoasttrain=np.zeros((251,36,23))
imgcoasttrain=dataloader(r'PRML_Assignment3_data\coast\train',imgcoasttrain)

imgforesttrain=np.zeros((229,36,23))
imgforesttrain=dataloader(r'PRML_Assignment3_data\forest\train',imgforesttrain)

imghighwaytrain=np.zeros((182,36,23))
imghighwaytrain=dataloader(r'PRML_Assignment3_data\highway\train',imghighwaytrain)

imgmountaintrain=np.zeros((261,36,23))
imgmountaintrain=dataloader(r'PRML_Assignment3_data\mountain\train',imgmountaintrain)

imgopencountrytrain=np.zeros((287,36,23))
imgopencountrytrain=dataloader(r'PRML_Assignment3_data\opencountry\train',imgopencountrytrain)


# In[ ]:


#Now finding the GMMs for the images

#function that loads the data but 36*23=828
def dataloader828(folder_path,loadeddata):
    #folder_path = r'PRML_Assignment3_data\coast\dev'
    i=0
    for filename in os.listdir(folder_path):
        with open(os.path.join(folder_path, filename), 'r') as f:
            text = f.read()
            temp=(list(map(str, text.strip().split("\n"))))
            for line in temp:
                temp1=(list(map(str, line.strip().split(","))))
                for line2 in temp1:
                    temp2=(list(map(float, line2.strip().split(" "))))
                    loadeddata[i,:]=temp2[:]
                    i+=1
    return(loadeddata)


# In[ ]:


#train data 828
imgcoasttrain828=np.zeros((251*36,23))
imgcoasttrain828=dataloader828(r'PRML_Assignment3_data\coast\train',imgcoasttrain828)

imgforesttrain828=np.zeros((229*36,23))
imgforesttrain828=dataloader828(r'PRML_Assignment3_data\forest\train',imgforesttrain828)

imghighwaytrain828=np.zeros((182*36,23))
imghighwaytrain828=dataloader828(r'PRML_Assignment3_data\highway\train',imghighwaytrain828)

imgmountaintrain828=np.zeros((261*36,23))
imgmountaintrain828=dataloader828(r'PRML_Assignment3_data\mountain\train',imgmountaintrain828)

imgopencountrytrain828=np.zeros((287*36,23))
imgopencountrytrain828=dataloader828(r'PRML_Assignment3_data\opencountry\train',imgopencountrytrain828)

imgtrain828=[imgcoasttrain828,imgforesttrain828,imghighwaytrain828,imgmountaintrain828,imgopencountrytrain828]


# In[ ]:


#normalising the data
'''
imgtrain828normalised=[]
for i in imgtrain828:
    min_max_scaler = preprocessing.MinMaxScaler()
    temp = min_max_scaler.fit_transform(i)
    imgtrain828normalised.append(temp)
#imgtrain828normalised
'''


# In[ ]:


K=5
#coast forest highway mountain opencountry

imgcoasttrainmeans,imgcoasttrainpoints,imgcoasttraingammas=Kmeans(K, imgcoasttrain828)
imgcoasttrainmeans,imgcoasttraincov,imgcoasttraingammas=GMMs(K, imgcoasttrain828, imgcoasttrainmeans, imgcoasttrainpoints, imgcoasttraingammas)
imgcoasttrainmeansd,imgcoasttraincovd,imgcoasttraingammasd=GMMscovdiag(K, imgcoasttrain828, imgcoasttrainmeans, imgcoasttrainpoints, imgcoasttraingammas)


# In[ ]:


imgforesttrainmeans,imgforesttrainpoints,imgforesttraingammas=Kmeans(K, imgforesttrain828)
imgforesttrainmeans,imgforesttraincov,imgforesttraingammas=GMMs(K, imgforesttrain828, imgforesttrainmeans, imgforesttrainpoints, imgforesttraingammas)
imgforesttrainmeansd,imgforesttraincovd,imgforesttraingammasd=GMMscovdiag(K, imgforesttrain828, imgforesttrainmeans, imgforesttrainpoints, imgforesttraingammas)


# In[ ]:


imghighwaytrainmeans,imghighwaytrainpoints,imghighwaytraingammas=Kmeans(K, imghighwaytrain828)
imghighwaytrainmeans,imghighwaytraincov,imghighwaytraingammas=GMMs(K, imghighwaytrain828, imghighwaytrainmeans, imghighwaytrainpoints, imghighwaytraingammas)
imghighwaytrainmeansd,imghighwaytraincovd,imghighwaytraingammasd=GMMscovdiag(K, imghighwaytrain828, imghighwaytrainmeans, imghighwaytrainpoints, imghighwaytraingammas)


# In[ ]:


imgmountaintrainmeans,imgmountaintrainpoints,imgmountaintraingammas=Kmeans(K, imgmountaintrain828)
imgmountaintrainmeans,imgmountaintraincov,imgmountaintraingammas=GMMs(K, imgmountaintrain828, imgmountaintrainmeans, imgmountaintrainpoints, imgmountaintraingammas)
imgmountaintrainmeansd,imgmountaintraincovd,imgmountaintraingammasd=GMMscovdiag(K, imgmountaintrain828, imgmountaintrainmeans, imgmountaintrainpoints, imgmountaintraingammas)


# In[ ]:


imgopencountrytrainmeans,imgopencountrytrainpoints,imgopencountrytraingammas=Kmeans(K, imgopencountrytrain828)
imgopencountrytrainmeans,imgopencountrytraincov,imgopencountrytraingammas=GMMs(K, imgopencountrytrain828, imgopencountrytrainmeans, imgopencountrytrainpoints, imgopencountrytraingammas)
imgopencountrytrainmeansd,imgopencountrytraincovd,imgopencountrytraingammasd=GMMscovdiag(K, imgopencountrytrain828, imgopencountrytrainmeans, imgopencountrytrainpoints, imgopencountrytraingammas)


# In[ ]:


allpies=np.zeros((5,K))
allpiesd=np.zeros((5,K))
#nanlocation=[]

for k in range(K):
    allpies[0][k]=np.sum(imgcoasttraingammas[:,k])
    allpies[1][k]=np.sum(imgforesttraingammas[:,k])
    allpies[2][k]=np.sum(imghighwaytraingammas[:,k])
    allpies[3][k]=np.sum(imgmountaintraingammas[:,k])
    allpies[4][k]=np.sum(imgopencountrytraingammas[:,k])
    
    allpiesd[0][k]=np.sum(imgcoasttraingammasd[:,k])
    allpiesd[1][k]=np.sum(imgforesttraingammasd[:,k])
    allpiesd[2][k]=np.sum(imghighwaytraingammasd[:,k])
    allpiesd[3][k]=np.sum(imgmountaintraingammasd[:,k])
    allpiesd[4][k]=np.sum(imgopencountrytraingammasd[:,k])
for i in range(5):
    #for j in range(len(allpies[0])):
        #if np.isnan(allpies[i][j])==True:
            #allpies[i][j]=0
            #nanlocation.append(i)
    allpies[i]=allpies[i]/np.sum(allpies[i])
    allpiesd[i]=allpiesd[i]/np.sum(allpiesd[i])
#nanlocation=list(set(nanlocation))


# In[ ]:


alltrainmeans=np.zeros((5,K,23))
alltraincov=np.zeros((5,K,23,23))

#means
alltrainmeans[0]=imgcoasttrainmeans
alltrainmeans[1]=imgforesttrainmeans
alltrainmeans[2]=imghighwaytrainmeans
alltrainmeans[3]=imgmountaintrainmeans
alltrainmeans[4]=imgopencountrytrainmeans

#covariances
alltraincov[0]=imgcoasttraincov
alltraincov[1]=imgforesttraincov
alltraincov[2]=imghighwaytraincov
alltraincov[3]=imgmountaintraincov
alltraincov[4]=imgopencountrytraincov


# In[ ]:


alltrainmeansd=np.zeros((5,K,23))
alltraincovd=np.zeros((5,K,23,23))

#means
alltrainmeansd[0]=imgcoasttrainmeansd
alltrainmeansd[1]=imgforesttrainmeansd
alltrainmeansd[2]=imghighwaytrainmeansd
alltrainmeansd[3]=imgmountaintrainmeansd
alltrainmeansd[4]=imgopencountrytrainmeansd

#covariances
alltraincovd[0]=imgcoasttraincovd
alltraincovd[1]=imgforesttraincovd
alltraincovd[2]=imghighwaytraincovd
alltraincovd[3]=imgmountaintraincovd
alltraincovd[4]=imgopencountrytraincovd


# In[ ]:


'''
#removing nans

f=len(nanlocation)
alltrainmeansn=np.zeros((5,K,23))
alltraincovn=np.zeros((5,K,23,23))
allpiesn=np.zeros((5,K))

for j in nanlocation:
    if nanlocation!=[]:
        alltraincovn[j[0]][j[1]]=alltraincov[j[0]][j[1]-1 if j[1]>0 else j[1]+1]
        alltrainmeansn[j[0]][j[1]]=alltrainmeans[j[0]][j[1]-1 if j[1]>0 else j[1]+1]
        allpiesn[j[0]][j[1]]=allpies[j[0]][j[1]-1 if j[1]>0 else j[1]+1]
    else:
        pass
'''


# In[ ]:


# classifying the points:

def classifier(K,datapoint,alltrainmeans,alltraincov,allpies):
    c=0
    numblocks=36
    temp3=-1
    temp1=0
    prob5classes=np.zeros(5)
    for c in range(5):
        temp2=0
        for a in range(numblocks):
            temp2 += np.log(np.sum([allpies[c][j]*np.exp(-0.5*((datapoint[a]-alltrainmeans[c][j]).T@np.linalg.pinv(alltraincov[c][j])@(datapoint[a]-alltrainmeans[c][j])))/(np.linalg.det(alltraincov[c][j]))**0.5 for j in range(K)]))
        #print(temp2)
        if temp2>temp1:
            temp1 = temp2 
            temp3 = c
        prob5classes[c]=temp2
    prob5classes=prob5classes/np.sum(prob5classes)
    return([temp3,prob5classes])


# In[ ]:


def pointsclassifierloader(K, devdata,cm,j,alltrainmeans,alltraincov,allpies):
    temp=[0]*len(devdata)
    for i in range(len(devdata)):
        temp[i]=classifier(K, devdata[i],alltrainmeans,alltraincov,allpies)[0]
    for i in temp:
        cm[j][i]+=1
    print(cm[j])


# In[ ]:


K=5
cm=np.zeros((5,5))
cmd=np.zeros((5,5))
pointsclassifierloader(K, imgcoastdev,cm,0,alltrainmeans,alltraincov,allpies)
pointsclassifierloader(K, imgcoastdev,cmd,0,alltrainmeansd,alltraincovd,allpiesd)


# In[ ]:


pointsclassifierloader(K, imgforestdev,cm,1,alltrainmeans,alltraincov,allpies)
pointsclassifierloader(K, imgforestdev,cmd,1,alltrainmeansd,alltraincovd,allpiesd)


# In[ ]:


pointsclassifierloader(K, imghighwaydev,cm,2,alltrainmeans,alltraincov,allpies)
pointsclassifierloader(K, imghighwaydev,cmd,2,alltrainmeansd,alltraincovd,allpiesd)


# In[ ]:


pointsclassifierloader(K, imgmountaindev,cm,3,alltrainmeans,alltraincov,allpies)
pointsclassifierloader(K, imgmountaindev,cmd,3,alltrainmeansd,alltraincovd,allpiesd)


# In[ ]:


pointsclassifierloader(K, imgopencountrydev,cm,4,alltrainmeans,alltraincov,allpies)
pointsclassifierloader(K, imgopencountrydev,cmd,4,alltrainmeansd,alltraincovd,allpiesd)


# In[ ]:


#the confusion matrix

def plot_matrix(cm, classes, title):
  ax = sn.heatmap(cm, cmap="Blues", annot=True, xticklabels=classes, yticklabels=classes, cbar=False)
  ax.set(title=title, xlabel="predicted label", ylabel="true label")

classes = ['coast', 'forest', 'highway', 'mountain', 'opencountry']
title = "Confusion Matrix for a full covariance matrix"
plot_matrix(cm, classes, title)
plt.show()
title = "Confusion Matrix for a diagonal covariance matrix"
plot_matrix(cmd, classes, title)
plt.show()


# In[ ]:


TP=0
Total=0

for i in range(5):
    for j in range(5):
        if i==j:
            TP+=cm[i][j]
        Total+=cm[i][j]
    
accuracy=(TP/Total)*100
print("accuracy using the normal method is: "+str(accuracy))


# In[ ]:


TPd=0
Totald=0

for i in range(5):
    for j in range(5):
        if i==j:
            TPd+=cmd[i][j]
        Totald+=cmd[i][j]
    
accuracy=(TPd/Totald)*100
print("accuracy using diagonal covariances is: "+str(accuracy))


# In[ ]:


def gmmaccuracyK(K,accstorer):

    imgcoasttrainmeans,imgcoasttrainpoints,imgcoasttraingammas=Kmeans(K, imgcoasttrain828)
    imgcoasttrainmeans,imgcoasttraincov,imgcoasttraingammas=GMMs(K, imgcoasttrain828, imgcoasttrainmeans, imgcoasttrainpoints, imgcoasttraingammas) 
    
    imgforesttrainmeans,imgforesttrainpoints,imgforesttraingammas=Kmeans(K, imgforesttrain828)
    imgforesttrainmeans,imgforesttraincov,imgforesttraingammas=GMMs(K, imgforesttrain828, imgforesttrainmeans, imgforesttrainpoints, imgforesttraingammas)

    imghighwaytrainmeans,imghighwaytrainpoints,imghighwaytraingammas=Kmeans(K, imghighwaytrain828)
    imghighwaytrainmeans,imghighwaytraincov,imghighwaytraingammas=GMMs(K, imghighwaytrain828, imghighwaytrainmeans, imghighwaytrainpoints, imghighwaytraingammas)

    imgmountaintrainmeans,imgmountaintrainpoints,imgmountaintraingammas=Kmeans(K, imgmountaintrain828)
    imgmountaintrainmeans,imgmountaintraincov,imgmountaintraingammas=GMMs(K, imgmountaintrain828, imgmountaintrainmeans, imgmountaintrainpoints, imgmountaintraingammas)

    imgopencountrytrainmeans,imgopencountrytrainpoints,imgopencountrytraingammas=Kmeans(K, imgopencountrytrain828)
    imgopencountrytrainmeans,imgopencountrytraincov,imgopencountrytraingammas=GMMs(K, imgopencountrytrain828, imgopencountrytrainmeans, imgopencountrytrainpoints, imgopencountrytraingammas)

    
    allpies=np.zeros((5,K))
    #nanlocation=[]

    for k in range(K):
        allpies[0][k]=np.sum(imgcoasttraingammas[:,k])
        allpies[1][k]=np.sum(imgforesttraingammas[:,k])
        allpies[2][k]=np.sum(imghighwaytraingammas[:,k])
        allpies[3][k]=np.sum(imgmountaintraingammas[:,k])
        allpies[4][k]=np.sum(imgopencountrytraingammas[:,k])
        
    for i in range(5):
        #for j in range(len(allpies[0])):
            #if np.isnan(allpies[i][j])==True:
                #allpies[i][j]=0
                #nanlocation.append(i)
        allpies[i]=allpies[i]/np.sum(allpies[i])
    #nanlocation=list(set(nanlocation))

    alltrainmeans=np.zeros((5,K,23))
    alltraincov=np.zeros((5,K,23,23))

    #means
    alltrainmeans[0]=imgcoasttrainmeans
    alltrainmeans[1]=imgforesttrainmeans
    alltrainmeans[2]=imghighwaytrainmeans
    alltrainmeans[3]=imgmountaintrainmeans
    alltrainmeans[4]=imgopencountrytrainmeans

    #covariances
    alltraincov[0]=imgcoasttraincov
    alltraincov[1]=imgforesttraincov
    alltraincov[2]=imghighwaytraincov
    alltraincov[3]=imgmountaintraincov
    alltraincov[4]=imgopencountrytraincov

    cm=np.zeros((5,5))
    pointsclassifierloader(K, imgcoastdev,cm,0,alltrainmeans,alltraincov,allpies)

    pointsclassifierloader(K, imgforestdev,cm,1,alltrainmeans,alltraincov,allpies)

    pointsclassifierloader(K, imghighwaydev,cm,2,alltrainmeans,alltraincov,allpies)

    pointsclassifierloader(K, imgmountaindev,cm,3,alltrainmeans,alltraincov,allpies)

    pointsclassifierloader(K, imgopencountrydev,cm,4,alltrainmeans,alltraincov,allpies)
    
    #the confusion matrix

    def plot_matrix(cm, classes, title):
      ax = sn.heatmap(cm, cmap="Blues", annot=True, xticklabels=classes, yticklabels=classes, cbar=False)
      ax.set(title=title, xlabel="predicted label", ylabel="true label")

    classes = ['coast', 'forest', 'highway', 'mountain', 'opencountry']
    title = "Confusion Matrix"
    
    plt.figure()
    plot_matrix(cm, classes, title)
    plt.title("confusion matrix of the images with K="+str(K))
    plt.show()

    TP=0
    Total=0

    for i in range(5):
        for j in range(5):
            if i==j:
                TP+=cm[i][j]
            Total+=cm[i][j]

    accuracy=(TP/Total)*100
    print(accuracy)
    
    accstorer.append([K,accuracy])


# In[ ]:


#getting the scores for the ROC
#coast forest highway mountain opencountry


scoresimg=[]
for i in imgcoastdev:
    temp1,temp2=classifier(K,i,alltrainmeans,alltraincov,allpies)
    scoresimg.append((temp2[0],0,temp1))
    scoresimg.append((temp2[1],1,temp1))
    scoresimg.append((temp2[2],2,temp1))
    scoresimg.append((temp2[3],3,temp1))
    scoresimg.append((temp2[4],4,temp1))
for i in imgforestdev:
    temp1,temp2=classifier(K,i,alltrainmeans,alltraincov,allpies)
    scoresimg.append((temp2[0],0,temp1))
    scoresimg.append((temp2[1],1,temp1))
    scoresimg.append((temp2[2],2,temp1))
    scoresimg.append((temp2[3],3,temp1))
    scoresimg.append((temp2[4],4,temp1))
for i in imghighwaydev:
    temp1,temp2=classifier(K,i,alltrainmeans,alltraincov,allpies)
    scoresimg.append((temp2[0],0,temp1))
    scoresimg.append((temp2[1],1,temp1))
    scoresimg.append((temp2[2],2,temp1))
    scoresimg.append((temp2[3],3,temp1))
    scoresimg.append((temp2[4],4,temp1))
for i in imgmountaindev:
    temp1,temp2=classifier(K,i,alltrainmeans,alltraincov,allpies)
    scoresimg.append((temp2[0],0,temp1))
    scoresimg.append((temp2[1],1,temp1))
    scoresimg.append((temp2[2],2,temp1))
    scoresimg.append((temp2[3],3,temp1))
    scoresimg.append((temp2[4],4,temp1))
for i in imgopencountrydev:
    temp1,temp2=classifier(K,i,alltrainmeans,alltraincov,allpies)
    scoresimg.append((temp2[0],0,temp1))
    scoresimg.append((temp2[1],1,temp1))
    scoresimg.append((temp2[2],2,temp1))
    scoresimg.append((temp2[3],3,temp1))
    scoresimg.append((temp2[4],4,temp1))


# In[ ]:


#getting the scores for the ROC for the diagonal matrix
#coast forest highway mountain opencountry


scoresimgd=[]
for i in imgcoastdev:
    temp1,temp2=classifier(K,i,alltrainmeansd,alltraincovd,allpiesd)
    scoresimgd.append((temp2[0],0,temp1))
    scoresimgd.append((temp2[1],1,temp1))
    scoresimgd.append((temp2[2],2,temp1))
    scoresimgd.append((temp2[3],3,temp1))
    scoresimgd.append((temp2[4],4,temp1))
for i in imgforestdev:
    temp1,temp2=classifier(K,i,alltrainmeansd,alltraincovd,allpiesd)
    scoresimgd.append((temp2[0],0,temp1))
    scoresimgd.append((temp2[1],1,temp1))
    scoresimgd.append((temp2[2],2,temp1))
    scoresimgd.append((temp2[3],3,temp1))
    scoresimgd.append((temp2[4],4,temp1))
for i in imghighwaydev:
    temp1,temp2=classifier(K,i,alltrainmeansd,alltraincovd,allpiesd)
    scoresimgd.append((temp2[0],0,temp1))
    scoresimgd.append((temp2[1],1,temp1))
    scoresimgd.append((temp2[2],2,temp1))
    scoresimgd.append((temp2[3],3,temp1))
    scoresimgd.append((temp2[4],4,temp1))
for i in imgmountaindev:
    temp1,temp2=classifier(K,i,alltrainmeansd,alltraincovd,allpiesd)
    scoresimgd.append((temp2[0],0,temp1))
    scoresimgd.append((temp2[1],1,temp1))
    scoresimgd.append((temp2[2],2,temp1))
    scoresimgd.append((temp2[3],3,temp1))
    scoresimgd.append((temp2[4],4,temp1))
for i in imgopencountrydev:
    temp1,temp2=classifier(K,i,alltrainmeansd,alltraincovd,allpiesd)
    scoresimgd.append((temp2[0],0,temp1))
    scoresimgd.append((temp2[1],1,temp1))
    scoresimgd.append((temp2[2],2,temp1))
    scoresimgd.append((temp2[3],3,temp1))
    scoresimgd.append((temp2[4],4,temp1))


# In[ ]:


plotROC(scoresimg)
plotDET(scoresimg)


# In[ ]:


plotROC(scoresimgd)
plotDET(scoresimgd)


# In[ ]:


#plotting the ROC

def plotmultiROC(scores1,scores2):
    plt.figure()
    c=0
    for scores in [scores1,scores2]:
        #fig,axis = plt.subplots(1,figsize=(10,8))
        temp_list = []
        for score in scores : 
            temp_list.append(score[0])
        scores = sorted(scores)
        temp_list = sorted(temp_list)
        TPR = []  # TP/(TP+FN)
        FNR = []  # FN/(FN+TP)
        FPR = []  # FP/(FP+TN)

        for threshold in temp_list : 
            tp = 0
            tn = 0
            fp = 0
            fn = 0 
            for data in scores : 
                if(data[0] >= threshold) : 
                    if(data[1] == data[2]) : 
                        tp += 1
                    else : 
                        fp += 1
                else : 
                    if(data[1] == data[2]) : 
                        fn += 1
                    else :
                        tn += 1
            TPR.append(float(tp/(tp+fn)))
            FNR.append(float(fn/(fn+tp)))
            FPR.append(float(fp/(fp+tn)))
        
#     display=sklearn.metrics.DetCurveDisplay(FPR, FNR, estimator_name=None, pos_label=None)
#     display.plot()
#     plt.show()
        if c==0:
            plt.plot(FPR,TPR, label="full covariance")
        if c==1:
            plt.plot(FPR,TPR, label="diagonal covariance")
        plt.title(f"ROC Curve")
        plt.xlabel("False Positive Rate(FPR)")
        plt.ylabel("True Positive Rate(TPR)")
        x=np.linspace(0,1,100)
        y=np.linspace(0,1,100)
        plt.plot(x,y)
        c+=1
    plt.legend()
    plt.show()


# In[ ]:


plotmultiROC(scoresimg,scoresimgd)


# In[ ]:


#plotting the DETs

def plotmultiDET(scores1,scores2): 
    plt.figure()
    c=0
    for scores in [scores1,scores2]:
        temp_list = []
        for score in scores : 
            temp_list.append(score[0])
        scores = sorted(scores)
        temp_list = sorted(temp_list)
        TPR = []  # TP/(TP+FN)
        FNR = []  # FN/(FN+TP)
        FPR = []  # FP/(FP+TN)

        for threshold in temp_list : 
            tp = 0
            tn = 0
            fp = 0
            fn = 0 
            for data in scores : 
                if(data[0] >= threshold) : 
                    if(data[1] == data[2]) : 
                        tp += 1
                    else : 
                        fp += 1
                else : 
                    if(data[1] == data[2]) : 
                        fn += 1
                    else :
                        tn += 1
            TPR.append(float(tp/(tp+fn)))
            FNR.append(float(fn/(fn+tp)))
            FPR.append(float(fp/(fp+tn)))

        blah1 = norm.ppf(FPR)
        blah2 = norm.ppf(FNR)
        if c==0:
            plt.plot(blah1,blah2, label='full covariance')
        if c==1:
            plt.plot(blah1,blah2, label='diagonal covariance')
        plt.title(f"DET Curve")
        plt.xlabel("False Alarm Rate")
        plt.ylabel("Missed Detection Rate")
        c+=1

    plt.legend()
    plt.show()


# In[ ]:


plotmultiDET(scoresimg,scoresimgd)


# In[ ]:


#comment this out if it takes too much time

# accstrer=[]
# for k in [2,3,4,5,6,7,8,9,10,12,15,17,20]:
#     gmmaccuracyK(k,accstrer)


# # In[ ]:


# #plotting accuracy vs K
# accstrer=np.array(accstrer)
# xaccstrer, yaccstrer = accstrer.T
# plt.scatter(xaccstrer,yaccstrer,color="black")
# plt.plot(accstrer[:,0],accstrer[:,1],color="purple")
# plt.grid()
# plt.xticks(np.arange(min(accstrer[:,0]), max(accstrer[:,0])+1, 1.0))
# plt.title("Accuracy of gmm for different values of K")
# plt.show()

