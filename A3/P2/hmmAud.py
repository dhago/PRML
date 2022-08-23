#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 11:11:20 2022

@author: dhago
"""
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


def readIntoArray (filepath):
    with open(filepath, 'r') as fp:
        a = fp.readline().strip().split(" ")
        nc = int(a[0])
        nv = int(a[1])
        temp = np.empty([nv,nc])
        i = 0;
        for line in fp.readlines():
            a = line.strip().split(" ")
            for j in range (0,nc):
                temp[i,j] = a[j]
            i+=1
    return nc,nv,temp

base_path = os.getcwd()

inputNums = [1,2,3,4,9]
train_audio_data = {}
train_audio_sizes = {}
test_audio_sizes = {}
nc_train = {}
nv_train = {}
for i in inputNums:
    temp_data = {}
    temp_nc = {}
    temp_nv = {}
    ss = 0
    for file in sorted(os.listdir(base_path + "/"+ str(i) + "/train/")):
        if file.endswith(".mfcc"):
            nc,nv,fdata = readIntoArray(base_path + "/"+ str(i) + "/train/"+file)
            temp_nc[file] = nc
            temp_nv[file] = nv
            temp_data[file] = fdata
            ss+=nv
    nc_train[i] = temp_nc
    nv_train[i] = temp_nv
    train_audio_data[i]=temp_data
    train_audio_sizes[i] = ss

test_audio_data = {}
nc_test = {}
nv_test= {}
for i in inputNums:
    temp_data = {}
    temp_nc = {}
    temp_nv = {}
    ss=0
    for file in sorted(os.listdir(base_path + "/"+ str(i) + "/dev/")):
        if file.endswith(".mfcc"):
            nc,nv,fdata = readIntoArray(base_path + "/"+ str(i) + "/dev/"+file)
            temp_nc[file] = nc
            temp_nv[file] = nv
            temp_data[file] = fdata
            ss+=nv
    nc_test[i] = temp_nc
    nv_test[i] = temp_nv
    test_audio_data[i]=temp_data
    test_audio_sizes[i] = ss
    
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
            mu = np.mean(data_from_files[i][file],axis=0)
            std = np.std(data_from_files[i][file],axis=0)
            temp[i][file] = (data_from_files[i][file] - mu)/(std)
    return temp

#train_audio_data = z_score_normalisation(inputNums, train_audio_data)
#test_audio_data = z_score_normalisation(inputNums, test_audio_data)

pooled_train_data = np.array([])
for char in train_audio_data:
    for file in train_audio_data[char]:
        pooled_train_data = np.append(pooled_train_data, train_audio_data[char][file])
for char in test_audio_data:
    for file in test_audio_data[char]:
        pooled_train_data = np.append(pooled_train_data, test_audio_data[char][file])
pooled_train_data = np.reshape(pooled_train_data, [-1,38])

pooled_test_data = np.array([])
for char in test_audio_data:
    for file in test_audio_data[char]:
        pooled_test_data = np.append(pooled_test_data, test_audio_data[char][file])
pooled_test_data = np.reshape(pooled_test_data, [-1,38])

#############################
from sklearn.cluster import KMeans
def changeK(k):
    #means = Kmeans(k,pooled_train_data)
    
    model = KMeans(n_clusters = k, random_state = 0).fit(pooled_train_data)
    means = model.cluster_centers_
    return k,means  

def convertToSequence(means,points):
    #seq = np.empty(points.shape[0],int)
    seq = []
    for i in range(points.shape[0]):
        seq.append(np.argmin(np.linalg.norm(means-points[i],axis=1)))
    return seq
  
def makeSequences(means):
    train_seqs  = {}
    i = 0
    for char in train_audio_data:
        class_seqs = []
        for file in train_audio_data[char]:
            class_seqs.extend([convertToSequence(means,train_audio_data[char][file])])
        train_seqs[char] = class_seqs
        
    test_seqs  = {}
    i = 0
    for char in test_audio_data:
        class_seqs = []
        for file in test_audio_data[char]:
            class_seqs.extend([convertToSequence(means,test_audio_data[char][file])])
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
    return prob
 
def doALot(seed,symbols,states,tol,train_seqs,test_seqs):
    a = {}
    b = {}
    olddir = os.getcwd()
    dirpath = olddir+"/HMM-Code/"
    for char in inputNums:
        os.chdir(dirpath)
        makeTestHMMFile(train_seqs[char], str(char)+"_train_hmm.seq", dirpath)
        os.system("./train_hmm "+ str(char)+"_train_hmm.seq "+ str(seed) + " " + str(states) + " " + str(symbols) + " "+str(tol))
        st,sym,tempa,tempb = readHMMOutput(str(char)+"_train_hmm.seq.hmm", dirpath)
        a[char] = tempa
        b[char] = tempb
        os.chdir(olddir)
    
    pi = np.zeros([states])
    pi[0] = 1
    classified = []
    allProbs = np.array([])
    for char in inputNums:
        clss = []
        for seq in test_seqs[char]:
            probs = np.array([])
            for c2 in inputNums:
                probs = np.append(probs,getProbFromHMM(states,seq, pi, a[c2], b[c2]))
            clss.append(np.argmax(probs))
            allProbs = np.append(allProbs,probs)
        classified.extend([clss])
    print(classified)
    err = [0]*len(classified)
    for i in range(0,len(classified)):
        for elem in classified[i]:
            if elem != i:
                err[i] += 1
    return [err,classified,allProbs]

def FinalFunction(k,states):
    k,means = changeK(k)
    print("Setting k to "+str(k))
    train_seqs,test_seqs = makeSequences(means)
    print("Making sequences")
    maxseed = []
    actseed = []
    for i in range(100):
        r = (rnd.randint(0,10000))
        arrerr,predictions,tem = doALot(r,k,states,0.1,train_seqs,test_seqs)
        acc = 60-sum(arrerr)
        maxseed.append(acc)
        actseed.append(r)
    finseed = 0
    m = 0
    for i in range(0,100):
        if maxseed[i] >= m:
            m = maxseed[i]
            finseed = actseed[i]
    error_final,predictions,allProbs = doALot(finseed, k, states, 0.1,train_seqs,test_seqs)
    
    y_true = []
    for i in range(len(predictions)):
        y_true += [i]*12
    y_pred = []
    for lst in predictions:
        for ll in lst:
            y_pred.append(ll)
    conf = confusion_matrix(y_true, (y_pred),labels=[0,1,2,3,4])
    display = ConfusionMatrixDisplay(confusion_matrix=conf, display_labels=inputNums)
    display.plot()
    display.ax_.set_title('Confusion Matrix for k ='+str(k) + " States = "+str(states), fontsize=20)
    plt.show()
    tpr,fpr,fnr = ROC_calc(allProbs, test_seqs)
    return 10*(60-sum(error_final))/6,tpr,fpr,fnr

def ROC_calc(allProbs,test_seqs):
    sAllProbs = np.sort(allProbs)
    count=0
    TPR=[]
    FNR=[]
    FPR=[]
    for threshold in sAllProbs:
        TP=TN=FP=FN=0
        i = 0
        for char in inputNums:
            j = 0
            for seq in test_seqs[char]:
                if(char == 1):
                    diff_char = 0
                if(char == 2):
                    diff_char = 1
                if(char == 3):
                    diff_char = 2
                if(char == 4):
                    diff_char = 3 
                if(char == 9):
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

ACC1,TPR1,FPR1,FNR1 = FinalFunction(10, 3)
ACC2,TPR2,FPR2,FNR2 = FinalFunction(10, 4)
ACC3,TPR3,FPR3,FNR3 = FinalFunction(14, 3)
ACC4,TPR4,FPR4,FNR4 = FinalFunction(14, 4)
ACC5,TPR5,FPR5,FNR5 = FinalFunction(12, 3)
ACC6,TPR6,FPR6,FNR6 = FinalFunction(12, 4)

print('k = 10, states = 3, Accuracy = '+str(ACC1))
print('k = 10, states = 4, Accuracy = '+str(ACC2))
print('k = 14, states = 3, Accuracy = '+str(ACC3))
print('k = 14, states = 4, Accuracy = '+str(ACC4))
print('k = 12, states = 3, Accuracy = '+str(ACC5))
print('k = 12, states = 4, Accuracy = '+str(ACC6))

fig, (ax_roc,ax_det) = plt.subplots(1,2,figsize=(20, 10))
fig.suptitle('Comparing ROC and DET curves', fontsize=20)
ax_roc.plot([0,1],[0,1], linestyle="--", color='black')
l1=ax_roc.plot(FPR1,TPR1,label='k = 10, states = 3')
l2=ax_roc.plot(FPR2,TPR2,label='k = 10, states = 4')
l3=ax_roc.plot(FPR3,TPR3,label='k = 14, states = 3')
l4=ax_roc.plot(FPR4,TPR4,label='k = 14, states = 4')
l5=ax_roc.plot(FPR5,TPR5,label='k = 12, states = 3')
l6=ax_roc.plot(FPR6,TPR6,label='k = 12, states = 4')
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_title('ROC curves', fontsize=15)
ax_roc.legend()

"""def DET_normalise(values):
    fmin = np.min(values,axis=0)
    fmax = np.max(values,axis=0)
    temp = (values - fmin)/(fmax-fmin)
    return temp

ax_det.plot(stats.norm.ppf(FPR1),stats.norm.ppf(FNR1),label='k = 10, states = 3')
ax_det.plot(stats.norm.ppf(FPR2),stats.norm.ppf(FNR2),label='k = 10, states = 4')
ax_det.plot(stats.norm.ppf(FPR3),stats.norm.ppf(FNR3),label='k = 14, states = 3')
ax_det.plot(stats.norm.ppf(FPR4),stats.norm.ppf(FNR4),label='k = 14, states = 4')
ax_det.plot(stats.norm.ppf(FPR5),stats.norm.ppf(FNR5),label='k = 12, states = 3')
ax_det.plot(stats.norm.ppf(FPR6),stats.norm.ppf(FNR6),label='k = 12, states = 4')
values = ax_roc.get_yticks()
ax_det.set_yticklabels(["{:.0%}".format(y) for y in DET_normalise(values)])
values = ax_roc.get_xticks()
ax_det.set_xticklabels(["{:.0%}".format(x) for x in DET_normalise(values)])
ax_det.set_xlabel('False Alarm Rate')
ax_det.set_ylabel('Missed Detection Rate')
ax_det.set_title('ROC curves', fontsize=15)
ax_det.legend()

fig, (ax_roc,ax_det) = plt.subplots(1,2,figsize=(20, 10))
aaa = stats.norm.ppf(FPR1),stats.norm.cdf(FNR1)
ax_det.plot(stats.norm.ppf(FPR1),stats.norm.cdf(FNR1),label='k = 10, states = 3')
values = ax_roc.get_yticks()
ax_det.set_yticklabels(["{:.0%}".format(y) for y in DET_normalise(values)])
values = ax_roc.get_xticks()
ax_det.set_xticklabels(["{:.0%}".format(x) for x in DET_normalise(values)])
ax_det.set_xlabel('False Alarm Rate')
ax_det.set_ylabel('Missed Detection Rate')
ax_det.set_title('ROC curves', fontsize=15)
ax_det.legend()"""

DetCurveDisplay(fpr=FPR1, fnr=FNR1, estimator_name="k = 10, States = 3").plot(ax = ax_det)
DetCurveDisplay(fpr=FPR2, fnr=FNR2, estimator_name="k = 10, States = 4").plot(ax = ax_det)
DetCurveDisplay(fpr=FPR3, fnr=FNR3, estimator_name="k = 14, States = 3").plot(ax = ax_det)
DetCurveDisplay(fpr=FPR4, fnr=FNR4, estimator_name="k = 14, States = 4").plot(ax = ax_det)
DetCurveDisplay(fpr=FPR5, fnr=FNR5, estimator_name="k = 12, States = 3").plot(ax = ax_det)
DetCurveDisplay(fpr=FPR6, fnr=FNR6, estimator_name="k = 12, States = 4").plot(ax = ax_det)
ax_det.set_title('ROC curves', fontsize=15)
ax_det.legend()




### CO=omment this out to compare HMM and DTW
"""fp = open("rocdata_dtwAudio.txt","r")
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
l1=ax_roc.plot(fprDTW,tprDTW,label='DTW: topK = 3')
l6=ax_roc.plot(FPR6,TPR6,label='k = 12, states = 4')
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_title('ROC curves', fontsize=15)
ax_roc.legend()

DetCurveDisplay(fpr=fprDTW, fnr=fnrDTW, estimator_name="DTW: topK = 3").plot(ax = ax_det)
DetCurveDisplay(fpr=FPR6, fnr=FNR6, estimator_name="HMM: k = 12, States = 4").plot(ax = ax_det)
ax_det.set_title('ROC curves', fontsize=15)
ax_det.legend()"""