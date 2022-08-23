#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 16:12:12 2022

@author: dhago
"""
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from sklearn.metrics import DetCurveDisplay, ConfusionMatrixDisplay, confusion_matrix

base_path = os.getcwd()
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
            
def readTeluguChars (filepath):
    with open(filepath, 'r') as fp:
        a = fp.readline().strip().split(" ") 
        size = int(a[0])
    return size,np.array(a[1:], 'float64').reshape([size,2])


telugu_chars = ["bA","chA","dA","lA","tA"]
tel_train_data = {}
tel_train_sizes = {}
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
        
tel_test_data = z_score_normalisation(telugu_chars, tel_test_data)
tel_train_data = z_score_normalisation(telugu_chars, tel_train_data)

"""xs = np.array([])
ys = np.array([])        
for file in tel_test_data["bA"]:
    print(file)
    xs = np.append(xs,tel_test_data["bA"][file][:,0])
    ys = np.append(ys,tel_test_data["bA"][file][:,1])

plt.scatter(xs,ys)
plt.show()"""

allProbs = {}
for char in telugu_chars:
    i = 0
    for file in tel_test_data[char]:
        test_array = tel_test_data[char][file]
        test_sz = tel_test_sizes[char][i]
        i+=1
        test_case_probs = []
        for char2 in telugu_chars: 
            each_class_prob = []
            j = 0
            for file2 in tel_train_data[char2]:
                train_array = tel_train_data[char2][file2]
                train_sz = tel_train_sizes[char2][j]
                j+=1
                dtw_val = dtw(test_array, test_sz, train_array, train_sz)
                each_class_prob.append(dtw_val)
            test_case_probs.append([each_class_prob])
            print("ONE FOLDER trains DONE")
        allProbs[char+file] = test_case_probs
        print("---------------"+char+ "_" + file+ " TEST DONE-----------------")

def varyTopK(topK):
    predicted_class2 = []
    errors2 = [0,0,0,0,0]
    wrong_files = []
    i = 0
    for char in telugu_chars:
        each_class_pred = []
        for file in tel_test_data[char]:
            j = 0
            predictor = np.empty((len(telugu_chars)))
            for lst in allProbs[char+file]:
                ll = lst[0]
                ll.sort()
                kAvg = sum(ll[0:topK])/topK
                predictor[j] = kAvg
                j+=1
            pred = np.argmin(predictor)
            each_class_pred.append(pred)
            if pred != i:
                wrong_files.append(char+"_"+file)
                errors2[i] += 1
        predicted_class2.append([each_class_pred])
        i+=1
    return errors2,predicted_class2,wrong_files

"""klist = []
errlist = []
for ii in range(1,31):
    a,b = varyTopK(ii)
    klist.append(ii)
    errlist.append(sum(a))"""
    
error2,predicted_class2,wfiles = varyTopK(2)
print("Errors:")
print(error2)
accur = 100*(100-sum(error2))/100
print("Accuracy = "+str(accur) +" for topK = "+str(2))
safe = allProbs
"""plt.scatter(tel_test_data["bA"]["73.txt"][:,0],tel_test_data["bA"]["73.txt"][:,1])
plt.show()
plt.scatter(tel_test_data["chA"]["81.txt"][:,0],tel_test_data["chA"]["81.txt"][:,1])
plt.show()"""

fp = open("dtwp2dump.txt",'w')
for char in telugu_chars:
    for file in tel_test_data[char]:
        fp.write(char+file+"\n")
        for lst in allProbs[char+file]:
            for item in lst[0]:
                fp.write(str(item)+" ")
        fp.write("\n")
fp.close()

y_true = []
for i in range(len(predicted_class2)):
    y_true += [i]*20
y_pred = []
for lst in predicted_class2:
    for ll in lst[0]:
        y_pred.append(ll)
conf = confusion_matrix(y_true, (y_pred),labels=[0,1,2,3,4])
display = ConfusionMatrixDisplay(confusion_matrix=conf, display_labels=telugu_chars)
display.plot()
plt.show()

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
        #k = 0
        for char in telugu_chars:
            for file in tel_test_data[char]:
                predictor = [0]*5
                j = 0
                for lst in allProbs[str(char)+file]:
                    ll = lst[0]
                    ll.sort()
                    kAvg = sum(ll[0:k])/k
                    predictor[j] = kAvg
                    j+=1
                pred = np.argmin(predictor)
                for i in range(5):
                    if predictor[i] <= threshold:
                        if i == pred:
                            TP+=1
                        else:
                            FP+=1
                    else:
                        if i == pred:
                            FN+=1
                        else:
                            TN+=1
                    
        #print("tot="+str(TP+FP+TN+FN))
        tpr.append(TP/(TP+FN))
        fpr.append(FP/(FP+TN))
        fnr.append(FN/(FN+TP))
        count+=1
    return tpr,fpr,fnr

tpr1,fpr1,fnr1 = ROC_calc(2)
tpr2,fpr2,fnr2 = ROC_calc(4)
tpr3,fpr3,fnr3 = ROC_calc(8)
tpr4,fpr4,fnr4 = ROC_calc(10)
tpr5,fpr5,fnr5 = ROC_calc(15)
fig, (ax_roc,ax_det) = plt.subplots(1,2,figsize=(20, 10))
fig.suptitle('Comparing DET and ROC for all cases', fontsize=20)
ax_roc.plot([0,1],[0,1], linestyle="--", color='black')
l1=ax_roc.plot(fpr1,tpr1,label='k = 2')
l2=ax_roc.plot(fpr2,tpr2,label='k = 4')
l3=ax_roc.plot(fpr3,tpr3,label='k = 8')
l4=ax_roc.plot(fpr4,tpr4,label='k = 10')
l5=ax_roc.plot(fpr5,tpr5,label='k = 15')
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_title('ROC curves', fontsize=15)
ax_roc.legend()

DetCurveDisplay(fpr=fpr1, fnr=fnr1, estimator_name="k = 2").plot(ax = ax_det)
DetCurveDisplay(fpr=fpr2, fnr=fnr2, estimator_name="k = 4").plot(ax = ax_det)
DetCurveDisplay(fpr=fpr3, fnr=fnr3, estimator_name="k = 8").plot(ax = ax_det)
DetCurveDisplay(fpr=fpr4, fnr=fnr4, estimator_name="k = 10").plot(ax = ax_det)
DetCurveDisplay(fpr=fpr5, fnr=fnr5, estimator_name="k = 15").plot(ax = ax_det)
ax_det.set_title('DET curve for different k', fontsize=20)
plt.legend()

fp = open("rocdata_dtwTel.txt","w")
for i in fpr1:
    fp.write(str(i)+" ")
fp.write("\n")
for i in tpr1:
    fp.write(str(i)+" ")
fp.write("\n")
for i in fnr1:
    fp.write(str(i)+" ")
fp.write("\n")
fp.close()