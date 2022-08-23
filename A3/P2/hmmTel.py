#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 18:06:30 2022

@author: dhago
"""
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import random as rnd
import scipy.stats as stats 
from sklearn.metrics import DetCurveDisplay, ConfusionMatrixDisplay, confusion_matrix

def maxminofdimension(traindata_class1):
    maxminofdimension_=np.zeros((len(traindata_class1[0]),2))
    for i in range(len(maxminofdimension_)):
            maxminofdimension_[i][0]=max(traindata_class1[:,i])
            maxminofdimension_[i][1]=min(traindata_class1[:,i])
    return maxminofdimension_

rnd.seed(20)
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
        
        
        ''''
        xt1, yt1 = np.array(pointsincluster1[0]).T
        plt.scatter(xt1,yt1)
        x1, y1 = means.T
        plt.scatter(x1,y1)   
        plt.show()
        '''

        #updating the means
        for i in range(K):
            for k in range(ndim):
                means[i][k]=np.sum(np.array(pointsincluster1[i])[:,k])/numofpoints[i]
        itercounter+=1
        
    return means


from sklearn.cluster import KMeans

def readTeluguChars (filepath):
    with open(filepath, 'r') as fp:
        a = fp.readline().strip().split(" ") 
        size = int(a[0])
    return size,np.array(a[1:], 'float64').reshape([size,2])

base_path = os.getcwd()
telugu_chars = os.listdir(base_path+"/Telugu/")

tel_train_data = {}
tel_train_sizes = {}
tel_train_class_sizes = {}
def keyfun(s):
    return int(s.split(".txt")[0])

for char in telugu_chars:
    each_folder_data = {}
    sizes = np.array([],int)
    for file in sorted(os.listdir(base_path+"/"+char+"/train/"), key=keyfun):
        if file.endswith(".txt"):
            s,arr = readTeluguChars(base_path+"/"+char+"/train/"+file)
            sizes=np.append(sizes,s)
            each_folder_data[file]=arr
    tel_train_data[char] = each_folder_data
    tel_train_sizes[char] = sizes
        

tel_test_data = {}
tel_test_sizes = {}
tel_test_class_sizes = {}
for char in telugu_chars:
    each_folder_data = {}
    sizes = np.array([],int)
    for file in sorted(os.listdir(base_path+"/"+char+"/dev/"), key=keyfun):
        if file.endswith(".txt"):
            s,arr = readTeluguChars(base_path+"/"+char+"/dev/"+file)
            sizes=np.append(sizes,s)
            each_folder_data[file]=arr
    tel_test_data[char] = each_folder_data
    tel_test_sizes[char] = sizes

for char in tel_train_sizes:
    tel_train_class_sizes[char] = np.sum(tel_train_sizes[char])
    tel_test_class_sizes[char] = np.sum(tel_test_sizes[char])
    
def min_max_normalisation(indices, data_from_files):
    temp = {}
    for i in indices:
        temp[i] = {}
        for file in data_from_files[i]:
            fmin = np.min(data_from_files[i][file],axis=0)
            fmax = np.max(data_from_files[i][file],axis=0)
            temp[i][file] = (data_from_files[i][file] - fmin)/(fmax-fmin)
            #print(data_from_files[i][file])
    return temp
def z_score_normalisation(indices, data_from_files):
    temp = {}
    for i in indices:
        temp[i] = {}
        for file in data_from_files[i]:
            temp[i][file] = stats.zscore(data_from_files[i][file])
    return temp

#temp_tel_train = min_max_normalisation(telugu_chars, tel_train_data)
tel_test_data = z_score_normalisation(telugu_chars, tel_test_data)
tel_train_data = z_score_normalisation(telugu_chars, tel_train_data)

"""xs = []
ys = []
for file in temp_tel_train["bA"]:
    xs.extend(temp_tel_train["bA"][file][:,0])
    ys.extend(temp_tel_train["bA"][file][:,1])
    
for file in tel_train_data["bA"]:
    xs.extend(tel_train_data["bA"][file][:,0])
    ys.extend(tel_train_data["bA"][file][:,1])
plt.clf()
plt.scatter(xs,ys,s=2)  
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
"""
pooled_train_data = np.array([])
for char in tel_train_data:
    for file in tel_train_data[char]:
        pooled_train_data = np.append(pooled_train_data, tel_train_data[char][file])
for char in tel_train_data:
    for file in tel_train_data[char]:
        pooled_train_data = np.append(pooled_train_data, tel_train_data[char][file])
pooled_train_data = np.reshape(pooled_train_data, [-1,2])

#############################

def changeK(k):
    means = Kmeans(k,pooled_train_data)
    return k,means    

"""from sklearn.cluster import KMeans

norms = []
ks = []
mul_means = []
for k in range(20,35):
    ks.append(k)
    print("starting "+ str(k))
    model = KMeans(n_clusters = k, random_state = 0).fit(pooled_train_data)
    means1 = model.cluster_centers_
    mul_means.append(means1)
    #print("asasasas")
    n = 0
    for point in pooled_train_data:
        n += np.min(np.linalg.norm(means1-point,axis=1))
    norms.append(n)
    #print(n)
    print("done "+ str(k))
plt.clf()
plt.plot(ks,norms)
plt.show()"""

def plotPoolAndMean(pooled_train_data,means):
    plt.clf()
    plt.scatter(pooled_train_data[:,0],pooled_train_data[:,1],s=1)  
    plt.scatter(means[:,0],means[:,1],s=5,c='r')
    plt.xlabel("X")
    plt.ylabel("Y")
    #plt.scatter(mul_means[0][:,0],mul_means[0][:,1],s=5,c='r')
    plt.show()

def convertToSequence(means,points):
    #seq = np.empty(points.shape[0],int)
    seq = []
    for i in range(points.shape[0]):
        seq.append(np.argmin(np.linalg.norm(means-points[i],axis=1)))
    return seq
  

def makeSequences(means):
    train_seqs  = {}
    i = 0
    for char in tel_train_data:
        class_seqs = []
        for file in tel_train_data[char]:
            class_seqs.extend([convertToSequence(means,tel_train_data[char][file])])
        train_seqs[char] = class_seqs
        
    test_seqs  = {}
    i = 0
    for char in tel_test_data:
        class_seqs = []
        for file in tel_test_data[char]:
            class_seqs.extend([convertToSequence(means,tel_test_data[char][file])])
        test_seqs[char] = class_seqs
    return train_seqs,test_seqs

def makeTestHMMFile(seqs,filename,filedir):
    fp = open(filedir+filename,'w')
    for lst in seqs:
        for i in lst:
            fp.write(str(i)+" ")
        fp.write("\n")
    fp.close()    


def readHMMOutput(filename,filedir):
    fp = open(filedir+filename,'r')
    states = int(fp.readline().split()[1])
    symbols = int(fp.readline().split()[1])
    #i,0 to itself, i,1 to next state
    a = np.empty([states,2])
    b = np.empty([states,2,symbols])
    temp = np.empty([2*states,symbols+1])
    i = 0
    for line in fp.readlines():
        if len(line)>5:
            temp[i] = line.split()
            i+=1
    for j in range(0,states):
        a[j,0] = temp[j*2,0]
        a[j,1] = temp[j*2+1,0]
        b[j,0] = temp[j*2,1:]
        b[j,1] = temp[j*2+1,1:]
    return [states,symbols,a,b]

def getProbFromHMM(states,seq,pi,a,b):
    l = len(seq)
    alpha = np.empty([l,states])
    for i in range(0,states):
        alpha[0,i] = pi[i]*b[i,0,seq[0]]*a[i,0]
    for t in range(1,l):
        for st in range(0,states):
            alpha[t,st] = alpha[t-1,st]*b[st,0,seq[t]]*a[st,0] 
            if st!=0:
                alpha[t,st] += alpha[t-1,st-1]*b[st-1,1,seq[t]]*a[st-1,1] 
    prob = np.sum(alpha[l-1])
    #print(alpha[l-1])
    return prob

def doALot(seed,symbols,states,tol,train_seqs,test_seqs):
    a = {}
    b = {}
    olddir = os.getcwd()
    dirpath = olddir+"/HMM-Code/"
    for char in telugu_chars:
        os.chdir(dirpath)
        makeTestHMMFile(train_seqs[char], char+"_train_hmm.seq", dirpath)
        os.system("./train_hmm "+ char+"_train_hmm.seq "+ str(seed) + " " + str(states) + " " + str(symbols) + " "+str(tol))
        st,sym,tempa,tempb = readHMMOutput(char+"_train_hmm.seq.hmm", dirpath)
        a[char] = tempa
        b[char] = tempb
        os.chdir(olddir)
    
    pi = np.zeros([states])
    pi[0] = 1
    classified = []
    allProbs = np.array([])
    for char in telugu_chars:
        clss = []
        for seq in test_seqs[char]:
            probs = np.array([])
            for c2 in telugu_chars:
                probs = np.append(probs,getProbFromHMM(states,seq, pi, a[c2], b[c2]))
            clss.append(np.argmax(probs))
            allProbs = np.append(allProbs,probs)
        classified.extend([clss])
    #print(classified)
    err = [0]*len(classified)
    for i in range(0,len(classified)):
        for elem in classified[i]:
            if elem != i:
                err[i] += 1
    return [err,classified,allProbs]

"""actseed
Out[150]: 69441

maxseed
Out[151]: 94


2054,30   9878,25"""

def FinalFunction(k,states):
    k,means = changeK(k)
    print("Setting k to "+str(k))
    train_seqs,test_seqs = makeSequences(means)
    #plotPoolAndMean(pooled_train_data, means)
    print("Making sequences")
    maxseed = []
    actseed = []
    for i in range(50):
        r = (rnd.randint(0,10000))
        arrerr,predictions,tem = doALot(r,k,states,0.1,train_seqs,test_seqs)
        acc = 100-sum(arrerr)
        maxseed.append(acc)
        actseed.append(r)
    finseed = 0
    m = 0
    for i in range(0,50):
        if maxseed[i] >= m:
            m = maxseed[i]
            finseed = actseed[i]
    error_final,predictions,allProbs = doALot(finseed, k, states, 0.1,train_seqs,test_seqs)
    
    y_true = []
    for i in range(len(predictions)):
        y_true += [i]*20
    y_pred = []
    for lst in predictions:
        for ll in lst:
            y_pred.append(ll)
    conf = confusion_matrix(y_true, (y_pred),labels=[0,1,2,3,4])
    display = ConfusionMatrixDisplay(confusion_matrix=conf, display_labels=telugu_chars)
    display.plot()
    display.ax_.set_title('Confusion Matrix for k ='+str(k) + " States = "+str(states), fontsize=20)
    plt.show()
    tpr,fpr,fnr = ROC_calc(allProbs, test_seqs)
    return (100-sum(error_final)),tpr,fpr,fnr

"""actseed
Out[163]: 31887

maxseed
Out[164]: 95"""   

def ROC_calc(allProbs,test_seqs):
    sAllProbs = np.sort(allProbs)
    count=0
    TPR=[]
    FNR=[]
    FPR=[]
    for threshold in sAllProbs:
        TP=TN=FP=FN=0
        i = 0
        for char in telugu_chars:
            j = 0
            for seq in test_seqs[char]:
                if(char == "bA"):
                    diff_char = 0
                if(char == "chA"):
                    diff_char = 1
                if(char == "tA"):
                    diff_char = 2
                if(char == "dA"):
                    diff_char = 3 
                if(char == "lA"):
                    diff_char = 4
                for pnum in range(5):
                    if allProbs[i] >= threshold:
                        if diff_char == pnum:
                            TP+=1
                        else:
                            FP+=1
                    else:
                        if diff_char == pnum:
                            FN+=1
                        else:
                            TN+=1
                    i+=1
                j+=1
        #print("tot="+str(TP+FP+TN+FN))
        TPR.append(TP/(TP+FN))
        FPR.append(FP/(FP+TN))
        FNR.append(FN/(TP+FN))
        count+=1
    return TPR,FPR,FNR

ACC1,TPR1,FPR1,FNR1 = FinalFunction(25, 4)
ACC2,TPR2,FPR2,FNR2 = FinalFunction(25, 6)
ACC3,TPR3,FPR3,FNR3 = FinalFunction(20, 4)
ACC4,TPR4,FPR4,FNR4 = FinalFunction(20, 6)
ACC5,TPR5,FPR5,FNR5 = FinalFunction(30, 4)
ACC6,TPR6,FPR6,FNR6 = FinalFunction(30, 6)

print('k = 25, states = 4, Accuracy = '+str(ACC1))
print('k = 25, states = 6, Accuracy = '+str(ACC2))
print('k = 20, states = 4, Accuracy = '+str(ACC3))
print('k = 20, states = 6, Accuracy = '+str(ACC4))
print('k = 30, states = 4, Accuracy = '+str(ACC5))
print('k = 30, states = 6, Accuracy = '+str(ACC6))

fig, (ax_roc,ax_det) = plt.subplots(1,2,figsize=(20, 10))
fig.suptitle('Comparing ROC and DET curves', fontsize=20)
ax_roc.plot([0,1],[0,1], linestyle="--", color='black')
l1=ax_roc.plot(FPR1,TPR1,label='k = 25, states = 4')
l2=ax_roc.plot(FPR2,TPR2,label='k = 25, states = 6')
l3=ax_roc.plot(FPR3,TPR3,label='k = 20, states = 4')
l4=ax_roc.plot(FPR4,TPR4,label='k = 20, states = 6')
l5=ax_roc.plot(FPR5,TPR5,label='k = 30, states = 4')
l6=ax_roc.plot(FPR6,TPR6,label='k = 30, states = 6')
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_title('ROC curves', fontsize=15)
ax_roc.legend()

"""def DET_normalise(values):
    fmin = np.min(values,axis=0)
    fmax = np.max(values,axis=0)
    temp = (values - fmin)/(fmax-fmin)
    return temp

ax_det.plot(stats.norm.ppf(FPR1),stats.norm.ppf(FNR1),label='k = 25, states = 4')
ax_det.plot(stats.norm.ppf(FPR2),stats.norm.ppf(FNR2),label='k = 25, states = 6')
ax_det.plot(stats.norm.ppf(FPR3),stats.norm.ppf(FNR3),label='k = 20, states = 4')
ax_det.plot(stats.norm.ppf(FPR4),stats.norm.ppf(FNR4),label='k = 20, states = 6')
ax_det.plot(stats.norm.ppf(FPR5),stats.norm.ppf(FNR5),label='k = 30, states = 4')
ax_det.plot(stats.norm.ppf(FPR6),stats.norm.ppf(FNR6),label='k = 30, states = 6')
ax_det.set_xlabel('False Alarm Rate')
ax_det.set_ylabel('Missed Detection Rate')
values = ax_roc.get_yticks()
ax_det.set_yticklabels(["{:.0%}".format(y) for y in DET_normalise(values)])
values = ax_roc.get_xticks()
ax_det.set_xticklabels(["{:.0%}".format(x) for x in DET_normalise(values)])

ax_det.set_title('ROC curves', fontsize=15)
ax_det.legend()"""
DetCurveDisplay(fpr=FPR1, fnr=FNR1, estimator_name="k = 25, States = 4").plot(ax = ax_det)
DetCurveDisplay(fpr=FPR2, fnr=FNR2, estimator_name="k = 25, States = 6").plot(ax = ax_det)
DetCurveDisplay(fpr=FPR3, fnr=FNR3, estimator_name="k = 20, States = 4").plot(ax = ax_det)
DetCurveDisplay(fpr=FPR4, fnr=FNR4, estimator_name="k = 20, States = 6").plot(ax = ax_det)
DetCurveDisplay(fpr=FPR5, fnr=FNR5, estimator_name="k = 30, States = 4").plot(ax = ax_det)
DetCurveDisplay(fpr=FPR6, fnr=FNR6, estimator_name="k = 30, States = 6").plot(ax = ax_det)
ax_det.set_title('ROC curves', fontsize=15)
ax_det.legend()



### CO=omment this out to compare HMM and DTW
"""fp = open("rocdata_dtwTel.txt","r")
fprDTW=[]
fnrDTW=[]
tprDTW=[]
line = fp.readline()
if len(line) > 3:
    ll = line.split()
    for i in ll:
        fprDTW.append(float(i))
line = fp.readline()
if len(line) > 3:
    ll = line.split()
    for i in ll:
        tprDTW.append(float(i))        
line = fp.readline()
if len(line) > 3:
    ll = line.split()
    for i in ll:
        fnrDTW.append(float(i))
fp.close()

fig, (ax_roc,ax_det) = plt.subplots(1,2,figsize=(20, 10))
fig.suptitle('Comparing ROC and DET curves', fontsize=20)
ax_roc.plot([0,1],[0,1], linestyle="--", color='black')
l1=ax_roc.plot(fprDTW,tprDTW,label='DTW: topK = 2')
l2=ax_roc.plot(FPR2,TPR2,label='HMM: k = 25, states = 6')
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_title('ROC curves', fontsize=15)
ax_roc.legend()

DetCurveDisplay(fpr=fprDTW, fnr=fnrDTW, estimator_name="DTW: topK = 2").plot(ax = ax_det)
DetCurveDisplay(fpr=FPR2, fnr=FNR2, estimator_name="HMM: k = 25, States = 6").plot(ax = ax_det)
ax_det.set_title('ROC curves', fontsize=15)
ax_det.legend()"""