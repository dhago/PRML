#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 15:13:24 2022

@author: dhago
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import DetCurveDisplay, ConfusionMatrixDisplay, confusion_matrix
import scipy.stats as stats 

def readAudioFile (filepath):
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

inputChars = [1,2,3,4,9]
base_path = os.getcwd()

train_audio_sizes = {}
test_audio_sizes = {}

train_audio_data = {}
nc_train = {}
nv_train = {}
for i in inputChars:
    temp_data = {}
    temp_nc = {}
    temp_nv = {}
    ss = 0
    for file in sorted(os.listdir(base_path + "/"+ str(i) + "/train/")):
        if file.endswith(".mfcc"):
            nc,nv,fdata = readAudioFile(base_path + "/"+ str(i) + "/train/"+file)
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
for i in inputChars:
    temp_data = {}
    temp_nc = {}
    temp_nv = {}
    ss=0
    for file in sorted(os.listdir(base_path + "/"+ str(i) + "/dev/")):
        if file.endswith(".mfcc"):
            nc,nv,fdata = readAudioFile(base_path + "/"+ str(i) + "/dev/"+file)
            temp_nc[file] = nc
            temp_nv[file] = nv
            temp_data[file] = fdata
            ss+=nv
    nc_test[i] = temp_nc
    nv_test[i] = temp_nv
    test_audio_data[i]=temp_data
    test_audio_sizes[i] = ss


def dist(x,y):
    return np.linalg.norm(x-y)

def dtw(a, len_a, b, len_b):
    dtw_matrix = np.empty((len_a+1, len_b+1))
    for i in range(len_a+1):
        dtw_matrix[i,0] = np.inf
    for j in range(len_b+1):
        dtw_matrix[0,j] = np.inf
    dtw_matrix[0, 0] = 0
    
    for i in range(1, len_a+1):
        for j in range(1, len_b+1):
            dtw_matrix[i, j] = dist(a[i-1],b[j-1]) + np.min([dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]])
    return dtw_matrix[len_a,len_b]

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

#test_audio_data = z_score_normalisation(inputChars, test_audio_data)
#train_audio_data = z_score_normalisation(inputChars, train_audio_data)
####### DO DTW 
answer = np.array([])
allProbs = {}
for char in inputChars:
    for file in test_audio_data[char]:
        test_array = test_audio_data[char][file]
        test_sz = nv_test[char][file]
        test_case_probs = []
        for char2 in inputChars: 
            each_class_prob = []
            for file2 in train_audio_data[char2]:
                train_array = train_audio_data[char2][file2]
                train_sz = nv_train[char2][file2]
                dtw_val = dtw(test_array, test_sz, train_array, train_sz)
                each_class_prob.append(dtw_val)
            test_case_probs.append([each_class_prob])
            print("ONE FOLDER trains DONE")
        allProbs[str(char)+file] = test_case_probs
        print("---------------"+str(char)+ "_" + file+ " TEST DONE-----------------")
   
topK = 3
predicted_class = []
errors = [0,0,0,0,0]
i = 0
p = 0
for char in inputChars:
    each_class_pred = []
    
    for file in test_audio_data[char]:
        j = 0
        predictor = np.empty((len(inputChars)))
        for lst in allProbs[str(char)+file]:
            ll = lst[0]
            ll.sort()
            kAvg = sum(ll[0:topK])/topK
            #print(kAvg)
            predictor[j] = kAvg
            j+=1
        pred = np.argmin(predictor)
        each_class_pred.append(pred)
        if pred != i:
            errors[i] += 1
    i+=1
    p+=1
    predicted_class.append([each_class_pred])
            
print("Errors:")
print(errors)
accur = 100*(60-sum(errors))/60
print("Accuracy = "+str(accur) +" for topK = "+str(topK))
safe = allProbs

fp = open("dtwp1dump.txt",'w')
for char in inputChars:
    for file in test_audio_data[char]:
        fp.write(str(char)+file+"\n")
        for lst in allProbs[str(char)+file]:
            for item in lst[0]:
                fp.write(str(item)+" ")
        fp.write("\n")
fp.close()


y_true = []
for i in range(len(predicted_class)):
    y_true += [i]*12
y_pred = []
for lst in predicted_class:
    for ll in lst[0]:
        y_pred.append(ll)
conf = confusion_matrix(y_true, (y_pred),labels=[0,1,2,3,4])
display = ConfusionMatrixDisplay(confusion_matrix=conf, display_labels=[1,2,3,4,9])
display.plot()
plt.show()

"""scores = []
i = 0
for char in inputChars:
    for file in test_audio_data[char]:
        temp = []
        for lst in allProbs[str(char)+file]:
            temp.extend(lst[0])
        scores.append((temp,y_true[i],y_pred[i]))"""
        
finAllProbs = []
for file in allProbs:
    for lst in allProbs[file]:
        #print(lst[0])
        finAllProbs.extend(lst[0])
#######################################
#####plotting roc
# WE vary topK
def ROC_calc(k):
    count=0
    tpr=[]
    fpr=[]
    fnr =[]
    templst = sorted(finAllProbs)
    for ind in range(0,len(templst),10):
        threshold = templst[ind]
        TP=TN=FP=FN=0
        for char in inputChars:
            for file in test_audio_data[char]:
                predictor = [0]*5
                j = 0
                for lst in allProbs[str(char)+file]:
                    ll = lst[0]
                    ll.sort()
                    kAvg = sum(ll[0:k])/k
                    predictor[j] = kAvg
                    j+=1
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
                for i in range(5):
                    if predictor[i] <= threshold:
                        if i == diff_char:
                            TP+=1
                        else:
                            FP+=1
                    else:
                        if i == diff_char:
                            FN+=1
                        else:
                            TN+=1
        #print("tot="+str(TP+FP+TN+FN))
        tpr.append(TP/(TP+FN))
        fpr.append(FP/(FP+TN))
        fnr.append(FN/(FN+TP))
        count+=1
    return tpr,fpr,fnr



tpr1,fpr1,fnr1 = ROC_calc(1)
tpr2,fpr2,fnr2 = ROC_calc(3)
tpr3,fpr3,fnr3 = ROC_calc(5)
tpr4,fpr4,fnr4 = ROC_calc(10)
tpr5,fpr5,fnr5 = ROC_calc(15)
fig, (ax_roc,ax_det) = plt.subplots(1,2,figsize=(20, 10))
fig.suptitle('Comparing DET and ROC for all cases', fontsize=20)
ax_roc.plot([0,1],[0,1], linestyle="--", color='black')
l1=ax_roc.plot(fpr1,tpr1,label='k = 1')
l2=ax_roc.plot(fpr2,tpr2,label='k = 3')
l3=ax_roc.plot(fpr3,tpr3,label='k = 5')
l4=ax_roc.plot(fpr4,tpr4,label='k = 10')
l5=ax_roc.plot(fpr5,tpr5,label='k = 15')
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_title('ROC curves', fontsize=15)
ax_roc.legend()


"""ax_det.plot(stats.norm.ppf(fpr1),stats.norm.ppf(fnr1),label='k = 10')
ax_det.plot(stats.norm.ppf(fpr2),stats.norm.ppf(fnr2),label='k = 3')
ax_det.plot(stats.norm.ppf(fpr3),stats.norm.ppf(fnr3),label='k = 5')
ax_det.plot(stats.norm.ppf(fpr4),stats.norm.ppf(fnr4),label='k = 10')
ax_det.plot(stats.norm.ppf(fpr5),stats.norm.ppf(fnr5),label='k = 15')

values = ax_roc.get_xticks()
ax_det.set_xticklabels(["{:.0%}".format(x) for x in DET_normalise(values)])
values = ax_roc.get_yticks()
ax_det.set_yticklabels(["{:.0%}".format(y) for y in DET_normalise(values)])
ax_det.set_xlabel('False Alarm Rate')
ax_det.set_ylabel('Missed Detection Rate')
ax_det.set_title('ROC curves', fontsize=15)
ax_det.legend()"""

DetCurveDisplay(fpr=fpr1, fnr=fnr1, estimator_name="k = 1").plot(ax = ax_det)
DetCurveDisplay(fpr=fpr2, fnr=fnr2, estimator_name="k = 3").plot(ax = ax_det)
DetCurveDisplay(fpr=fpr3, fnr=fnr3, estimator_name="k = 5").plot(ax = ax_det)
DetCurveDisplay(fpr=fpr4, fnr=fnr4, estimator_name="k = 10").plot(ax = ax_det)
DetCurveDisplay(fpr=fpr5, fnr=fnr5, estimator_name="k = 15").plot(ax = ax_det)
ax_det.set_title('DET curve for different k', fontsize=20)
plt.legend()

fp = open("rocdata_dtwAudio.txt","w")
for i in fpr2:
    fp.write(str(i)+" ")
fp.write("\n")
for i in tpr2:
    fp.write(str(i)+" ")
fp.write("\n")
for i in fnr2:
    fp.write(str(i)+" ")
fp.write("\n")
fp.close()







































