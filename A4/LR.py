#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 19:02:35 2022

@author: dhago
"""
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import random as rnd
import scipy.stats as stats
import scipy 
from sklearn.metrics import DetCurveDisplay, ConfusionMatrixDisplay, confusion_matrix


def ROC_calc(allProbs,num_classes,act_c):
    cc = len(allProbs)
    flat_list = [item for sublist in allProbs for item in sublist]
    sAllProbs = np.sort(np.array(flat_list))
    count=0
    TPR=[]
    FNR=[]
    FPR=[]
    for ll in range(0,sAllProbs.shape[0]):
        threshold = sAllProbs[ll]
        TP=TN=FP=FN=0
        acc = 0
        for i in range(cc):         
            for pnum in range(num_classes):
                if allProbs[i][pnum] >= threshold:
                    if act_c[i] == pnum:
                        TP+=1
                    else:
                        FP+=1
                else:
                    if act_c[i] == pnum:
                        FN+=1
                    else:
                        TN+=1
        TPR.append(TP/(TP+FN))
        FPR.append(FP/(FP+TN))
        FNR.append(FN/(TP+FN))
        count+=1
    return TPR,FPR,FNR

def confMatPlotter(allProbs,num_classes,act_c,class_labels,name):
    pred_c = []
    for i in allProbs:
        pred_c.append(np.argmax(i))
    conf = confusion_matrix(act_c, (pred_c),labels=[i for i in range(num_classes)])
    display = ConfusionMatrixDisplay(confusion_matrix=conf, display_labels=class_labels)
    display.plot()
    display.ax_.set_title('Confusion Matrix: '+name, fontsize=15)
    plt.show()
def plot_ROC_DET(probs1,probs2,probs3,num_classes,class_labels,name):
    tpr1_syn,fpr1_syn,fnr1_syn = ROC_calc(probs1, num_classes,class_labels)
    tpr2_syn,fpr2_syn,fnr2_syn = ROC_calc(probs2, num_classes,class_labels)
    tpr3_syn,fpr3_syn,fnr3_syn = ROC_calc(probs3, num_classes,class_labels)
    fig, (ax_roc,ax_det) = plt.subplots(1,2,figsize=(20, 10))
    fig.suptitle('ROC and DET for '+name, fontsize=20)
    ax_roc.plot([0,1],[0,1], linestyle="--", color='black')
    l1=ax_roc.plot(fpr1_syn,tpr1_syn,label='Without PCA/LDA')
    l2=ax_roc.plot(fpr2_syn,tpr2_syn,label='After PCA')
    l3=ax_roc.plot(fpr3_syn,tpr3_syn,label='After LDA')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC curves', fontsize=15)
    ax_roc.legend(loc=0, prop={'size': 20})
    DetCurveDisplay(fpr=fpr1_syn, fnr=fnr1_syn, estimator_name='Without PCA/LDA').plot(ax = ax_det)
    DetCurveDisplay(fpr=fpr2_syn, fnr=fnr2_syn, estimator_name='After PCA').plot(ax = ax_det)
    DetCurveDisplay(fpr=fpr3_syn, fnr=fnr3_syn, estimator_name='After LDA').plot(ax = ax_det)
    ax_det.set_title('DET curves', fontsize=15)
    ax_det.legend(loc=0, prop={'size': 20})  
    
def save_ROC(probs1,num_classes,class_labels,fp):
    tpr1_syn,fpr1_syn,fnr1_syn = ROC_calc(probs1, num_classes,class_labels)
    for i in tpr1_syn:
        fp.write(str(i)+" ")
    fp.write("\n")
    for i in fpr1_syn:
        fp.write(str(i)+" ")
    fp.write("\n")
base_path = os.getcwd()
syn_path = base_path+"/19/"

########## Read synthetic data
syn_train = []
syn_t_class = []

fp = open(syn_path+"train.txt",'r')
for line in fp:
    act_line = line.split()[0]
    nums = act_line.split(",")
    syn_train.append([float(nums[0]),float(nums[1])])
    syn_t_class.append(int(nums[2])-1)
  
syn_dev = []
syn_d_class = []
fp.close()
fp = open(syn_path+"dev.txt",'r')
for line in fp:
    act_line = line.split()[0]
    nums = act_line.split(",")
    syn_dev.append([float(nums[0]),float(nums[1])])
    syn_d_class.append(int(nums[2])-1)
fp.close()

########### Read image data
def dataloader828(folder_path,loadeddata):
    #folder_path = r'PRML_Assignment3_data\coast\dev'
    i=0
    for filename in os.listdir(folder_path):
        with open(os.path.join(folder_path, filename), 'r') as f:
            text = f.read()
            temp=(list(map(str, text.strip().split("\n"))))
            
            l=0
            for line in temp:
                temp1=(list(map(float, line.strip().split(" "))))
                if l ==0:
                    temp2 = temp1
                else:
                    temp2 = np.concatenate((temp2,temp1))
                l+=1 
            loadeddata[i,:]=temp2[:]
            i+=1
    return(loadeddata)
#train data 828
imgcoasttrain828=np.zeros((251,828))
imgcoasttrain828=dataloader828(base_path+'/coast/train',imgcoasttrain828)

imgforesttrain828=np.zeros((229,828))
imgforesttrain828=dataloader828(base_path+'/forest/train',imgforesttrain828)

imghighwaytrain828=np.zeros((182,828))
imghighwaytrain828=dataloader828(base_path+'/highway/train',imghighwaytrain828)

imgmountaintrain828=np.zeros((261,828))
imgmountaintrain828=dataloader828(base_path+'/mountain/train',imgmountaintrain828)

imgopencountrytrain828=np.zeros((287,828))
imgopencountrytrain828=dataloader828(base_path+'/opencountry/train',imgopencountrytrain828)

imgtrain828=[imgcoasttrain828,imgforesttrain828,imghighwaytrain828,imgmountaintrain828,imgopencountrytrain828]


#dev data 828
imgcoastdev828=np.zeros((73,828))
imgcoastdev828=dataloader828(base_path+'/coast/dev',imgcoastdev828)

imgforestdev828=np.zeros((66,828))
imgforestdev828=dataloader828(base_path+'/forest/dev',imgforestdev828)

imghighwaydev828=np.zeros((52,828))
imghighwaydev828=dataloader828(base_path+'/highway/dev',imghighwaydev828)

imgmountaindev828=np.zeros((75,828))
imgmountaindev828=dataloader828(base_path+'/mountain/dev',imgmountaindev828)

imgopencountrydev828=np.zeros((82,828))
imgopencountrydev828=dataloader828(base_path+'/opencountry/dev',imgopencountrydev828)

imgdev828=[imgcoastdev828,imgforestdev828,imghighwaydev828,imgmountaindev828,imgopencountrydev828]

def min_max_score_normalisation_v3(data):
    fmin = np.min(data,axis=0)
    fmax = np.max(data,axis=0)
    return (data - fmin)/(fmax-fmin)
img_train_pool = imgtrain828[0]
img_test_pool = imgdev828[0]
img_train_class = [0]*251+[1]*229+[2]*182+[3]*261+[4]*287
#251,229,182,261,287,73,66,52,75,82
img_test_class = [0]*73+[1]*66+[2]*52+[3]*75+[4]*82
for i in range(1,5):
    img_train_pool = np.concatenate((img_train_pool,min_max_score_normalisation_v3(imgtrain828[i])))
    img_test_pool = np.concatenate((img_test_pool,min_max_score_normalisation_v3(imgdev828[i])))
len(img_test_pool)
imgdev828 = [img_test_pool[0:73],img_test_pool[73:139],img_test_pool[139:191],img_test_pool[191:266],img_test_pool[266:348]]
imgtrain828 = [img_train_pool[0:251],img_train_pool[251:480],img_train_pool[480:662],img_train_pool[662:923],img_train_pool[923:1271]]
########### Read mfcc files
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

inputNums = [1,2,3,4,9]
aud_train = {}
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
    aud_train[i]=temp_data
    train_audio_sizes[i] = ss

aud_test = {}
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
    aud_test[i]=temp_data
    test_audio_sizes[i] = ss


########### Read Telugu data
def readTeluguChars (filepath):
    with open(filepath, 'r') as fp:
        a = fp.readline().strip().split(" ") 
        size = int(a[0])
    return size,np.array(a[1:], 'float64').reshape([size,2])


telugu_chars =  ['bA', 'chA', 'tA', 'dA', 'lA']
tel_train = {}
tel_train_sizes = {}
def keyfun(s):
    return int(s.split(".txt")[0])

for char in telugu_chars:
    each_folder_data = {}
    sizes = np.array([],int)
    temp_nv ={}
    for file in sorted(os.listdir(base_path+"/"+char+"/train/"), key=keyfun):
        if file.endswith(".txt"):
            s,arr = readTeluguChars(base_path+"/"+char+"/train/"+file)
            temp_nv[file] = s
            sizes=np.append(sizes,s)
            each_folder_data[file]=arr
    tel_train[char] = each_folder_data
    tel_train_sizes[char] = temp_nv
        

tel_test = {}
tel_test_sizes = {}
for char in telugu_chars:
    each_folder_data = {}
    sizes = np.array([],int)
    temp_nv ={}
    for file in sorted(os.listdir(base_path+"/"+char+"/dev/"), key=keyfun):
        if file.endswith(".txt"):
            s,arr = readTeluguChars(base_path+"/"+char+"/dev/"+file)
            temp_nv[file] = s
            sizes=np.append(sizes,s)
            each_folder_data[file]=arr
    tel_test[char] = each_folder_data
    tel_test_sizes[char] = temp_nv
def z_score_normalisation(indices, data_from_files):
    temp = {}
    for i in indices:
        temp[i] = {}
        for file in data_from_files[i]:
            temp[i][file] = stats.zscore(data_from_files[i][file])
    return temp
def min_max_normalisation_4(indices, data_from_files):
    temp = {}
    for i in indices:
        temp[i] = {}
        for file in data_from_files[i]:
            fmin = np.min(data_from_files[i][file],axis=0)
            fmax = np.max(data_from_files[i][file],axis=0)
            temp[i][file] = (data_from_files[i][file] - fmin)/(fmax-fmin)
            #print(data_from_files[i][file])
    return temp
#temp_tel_train = min_max_normalisation(telugu_chars, tel_train_data)
tel_test = z_score_normalisation(telugu_chars, tel_test)
tel_train = z_score_normalisation(telugu_chars, tel_train)


def ChangeLengthAndFlatten(train,train_nv,test,test_nv,chars):
    final_train = []
    train_c = []
    test_c = []
    final_test = []
    min_len =np.inf
    for char in chars:
        min1 = min(train_nv[char].values())
        min2 = min(test_nv[char].values())
        min1 = min(min2,min1)
        min_len = min(min1,min_len)
    for char in chars:
        temp1 = []
        temp2 = []
        cc = 0
        cc2 = 0
        for file in train[char]:
            cc+=1
            window = train_nv[char][file]-min_len+1
            x = []
            for i in range(min_len):
                avg = np.mean(train[char][file][i:i+window],axis=0).tolist()
                x.extend(avg)
            temp1.append(x)
        for file in test[char]:
            cc2+=1
            window = test_nv[char][file]-min_len+1
            x = []
            for i in range(min_len):
                avg = np.mean(test[char][file][i:i+window],axis=0).tolist()
                x.extend(avg)
            temp2.append(x)
        final_train.append(np.array(temp1))
        final_test.append(np.array(temp2))
        train_c.append(cc)
        test_c.append(cc2)
    return final_train,final_test,min_len,train_c,test_c

def zscore_normalisation_v2(data):
    return stats.zscore(data)
def min_max_normalisation_v2(data):
    fmin = np.min(data,axis=0)
    fmax = np.max(data,axis=0)
    return (data - fmin)/(fmax-fmin)

####### Normalise synthetic data 
syn_train = zscore_normalisation_v2(syn_train)
syn_dev = zscore_normalisation_v2(syn_dev)
syn_t_list = []
syn_t_list.append(syn_train[0:1250])
syn_t_list.append(syn_train[1250:2500])
####### Flatten audio and tel data
aud_train,aud_test,c,aud_train_count,aud_test_count=ChangeLengthAndFlatten(aud_train, nv_train, aud_test, nv_test, inputNums)
tel_train,tel_test,c,tel_train_count,tel_test_count=ChangeLengthAndFlatten(tel_train, tel_train_sizes, tel_test, tel_test_sizes, telugu_chars)


#######
aud_train_pool = []
aud_tr_pool_class = []
aud_test_pool = []
aud_te_pool_class = []
for char in inputNums:
    cc = 0
    if char == 2:
        cc = 1
    if char == 3:
        cc = 2
    if char == 4:
        cc = 3
    if char == 9:
        cc = 4
    for file in aud_train[cc]:
        aud_train_pool.append(file)
        aud_tr_pool_class.append(cc)
    for file in aud_test[cc]:
        aud_test_pool.append(file)
        aud_te_pool_class.append(cc)
aud_train_pool = zscore_normalisation_v2(aud_train_pool)
aud_test_pool= zscore_normalisation_v2(aud_test_pool)
########
tel_train_pool = []
tel_tr_pool_class = []
tel_test_pool = []
tel_te_pool_class = []
for char in telugu_chars:
    cc = 0
    if char == 'chA':
        cc = 1
    if char == 'tA':
        cc = 2
    if char == 'dA':
        cc = 3
    if char == 'lA':
        cc = 4
    for file in tel_train[cc]:
        tel_train_pool.append(file)
        tel_tr_pool_class.append(cc)
    for file in tel_test[cc]:
        tel_test_pool.append(file)
        tel_te_pool_class.append(cc)
tel_train_pool = zscore_normalisation_v2(tel_train_pool)
tel_test_pool = zscore_normalisation_v2(tel_test_pool)
############
def genPhiDeg1(data):
    final = [1]
    for x in data:
        final.append(x)
    return np.array(final)
def genPhiDeg2(data):
    final = [1]
    for i in range(len(data)):
        final.append(data[i])
        for j in range(len(data)):
            final.append(data[i]*data[j])
    return np.array(final)
def genPhiDeg3(data):
    final = [1]
    for i in range(len(data)):
        final.append(data[i])
        for j in range(i,len(data)):
            for k in range(j,len(data)):
                final.append(data[i]*data[j]*data[k])
    return np.array(final)

def sigmoid_prob(ws,x,k):
    summ = 0
    for i in range(len(ws)):
        summ += math.exp((np.transpose(ws[i])-np.transpose(ws[k]))@x)
    return 1/summ

def sigmoid_prob_max_index(ws,x):
    maxx = -np.inf
    ind = 0
    base = 0
    all_probs = []
    for i in range(len(ws)):
        base += math.exp((np.transpose(ws[i]))@x)
    for i in range(len(ws)):
        val = math.exp(np.transpose(ws[i])@x)
        all_probs.append(val/base)
        if maxx < val:
            ind = i
            maxx = val
    return ind,all_probs

def LogisticRegression(train,t_class,test,test_class,num_iter,eta,num_classes):
    phi_ex = genPhiDeg1(train[0])
    ws = [np.zeros([phi_ex.shape[0],1])]*num_classes
    #print(phi_ex.shape[0])
    for i in range(num_iter):
        summ = [np.zeros(ws[0].shape)]*num_classes
        for j in range(len(train)):
            x = genPhiDeg1(train[j])
            x = x.reshape(x.shape[0],1)
            for c in range(num_classes):
                y = 0
                if c == t_class[j]:
                    y = 1
                summ[c] = summ[c] + (sigmoid_prob(ws,x,c)-y)*(x)
        for c in range(num_classes):
            ws[c] = ws[c] - summ[c]*eta
    #print(ws)
    acc = 0
    all_probs = []
    for i in range(len(test)):
        x = genPhiDeg1(test[i])
        x = x.reshape(x.shape[0],1)
        ind,x_probs = sigmoid_prob_max_index(ws, x)
        all_probs.append(x_probs)
        if test_class[i]==ind:
            acc+=1   
            
    return 100*acc/len(test),all_probs

def LogisticRegressionV2(train,t_class,test,test_class,num_iter,eta,num_classes):
    phi_ex = genPhiDeg2(train[0])
    ws = [np.zeros([phi_ex.shape[0],1])]*num_classes
    #print(phi_ex.shape[0])
    for i in range(num_iter):
        summ = [np.zeros(ws[0].shape)]*num_classes
        for j in range(len(train)):
            x = genPhiDeg2(train[j])
            x = x.reshape(x.shape[0],1)
            for c in range(num_classes):
                y = 0
                if c == t_class[j]:
                    y = 1
                summ[c] = summ[c] + (sigmoid_prob(ws,x,c)-y)*(x)
        for c in range(num_classes):
            ws[c] = ws[c] - summ[c]*eta
    #print(ws)
    acc = 0
    all_probs = []
    for i in range(len(test)):
        x = genPhiDeg2(test[i])
        x = x.reshape(x.shape[0],1)
        ind,x_probs = sigmoid_prob_max_index(ws, x)
        all_probs.append(x_probs)
        if test_class[i]==ind:
            acc+=1   
            
    return 100*acc/len(test),all_probs

def PCAretQ(full_data, d):
    means = np.mean(full_data,axis=0)
    y = (full_data-means)
    cov = (np.transpose(y)@y)
    w,v = np.linalg.eigh(cov)
    sorted_i = np.argsort(-np.absolute(w))
    w = w[sorted_i]
    v = v[:,sorted_i[0:d]]
    return v
def LDAMultiClass(num_classes, data_list,d2):
    d = data_list[0].shape[1]
    summ = np.zeros([1,d])
    tot_size = 0
    final = []
    mu = []
    for i in range(num_classes):
        mu1 = np.array([np.mean(data_list[i],axis=0)])
        mu.append(mu1)
        summ += mu1*data_list[i].shape[0]
        tot_size += data_list[i].shape[0]
    mu_tot = summ/tot_size
    sb = np.zeros([d,d])
    sw = np.zeros([d,d])
    for i in range(num_classes):
        mu1 = mu[i]
        n = data_list[i].shape[0]
        sb += np.dot(np.transpose(mu1-mu_tot),(mu1-mu_tot))
        sw += np.cov(np.transpose(data_list[i]-mu1)@(data_list[i]-mu1),rowvar=False)

    qq = np.linalg.pinv(sw)@sb
    w,v = np.linalg.eigh(qq)
    sorted_i = np.argsort(-np.absolute(w))
    w = w[sorted_i]
    v = v[:,sorted_i[0:d2]]        
    return v

######### Synthetic data normal,pca,lda
syn_acc, syn_probs = LogisticRegression(syn_train, syn_t_class, syn_dev, syn_d_class,100, 0.00001,2)
print("Logistic Regression: Synthetic Data : Accuracy = "+str(syn_acc))
confMatPlotter(syn_probs, 2, syn_d_class, [1,2], "Synthetic Data")
vv = PCAretQ(syn_train, 1)
syn_pca_acc,syn_pca_prob = LogisticRegression(syn_train@vv, syn_t_class, syn_dev@vv, syn_d_class,100, 0.00001,2)
print("Logistic Regression: Synthetic Data(PCA) : Accuracy = "+str(syn_pca_acc))
confMatPlotter(syn_pca_prob, 2, syn_d_class, [1,2], "Synthetic Data(PCA)")
vv = LDAMultiClass(2, syn_t_list, 1)
syn_lda_acc,syn_lda_prob = LogisticRegression(syn_train@vv, syn_t_class, syn_dev@vv, syn_d_class,50, 0.00001,2)
print("Logistic Regression: Synthetic Data(LDA) : Accuracy = "+str(syn_lda_acc))
confMatPlotter(syn_lda_prob, 2, syn_d_class, [1,2], "Synthetic Data(LDA)")

plot_ROC_DET(syn_probs, syn_pca_prob, syn_lda_prob, 2, syn_d_class, "Synthetic Data")


########## Audio files
aud_acc, aud_probs = LogisticRegression(aud_train_pool, aud_tr_pool_class, aud_test_pool, aud_te_pool_class,50, 0.00001,5)
print("Logistic Regression: Audio Data : Accuracy = "+str(aud_acc))
confMatPlotter(aud_probs, 5, aud_te_pool_class, [1,2,3,4,9], "Audio Data")
vv = PCAretQ(aud_train_pool, 20)
aud_pca_acc, aud_pca_probs = LogisticRegression(aud_train_pool@vv, aud_tr_pool_class, aud_test_pool@vv, aud_te_pool_class,50, 0.00001,5)
print("Logistic Regression: Audio Data(PCA) : Accuracy = "+str(aud_pca_acc))
confMatPlotter(aud_pca_probs, 5, aud_te_pool_class, [1,2,3,4,9], "Audio Data(PCA)")
vv = LDAMultiClass(5,aud_train, 20)
aud_lda_acc, aud_lda_probs = LogisticRegression(aud_train_pool@vv, aud_tr_pool_class, aud_test_pool@vv, aud_te_pool_class,50, 0.00001,5)
print("Logistic Regression: Audio Data(LDA) : Accuracy = "+str(aud_lda_acc))
confMatPlotter(aud_lda_probs, 5, aud_te_pool_class, [1,2,3,4,9], "Audio Data(LDA)")

plot_ROC_DET(aud_probs, aud_pca_probs, aud_lda_probs, 5, aud_te_pool_class, "Audio Data")

########## Telugu files
tel_acc, tel_probs = LogisticRegression(tel_train_pool, tel_tr_pool_class, tel_test_pool, tel_te_pool_class,700, 0.001,5)
print("Logistic Regression: Handwritten Data 60: Accuracy = "+str(tel_acc))
confMatPlotter(tel_probs, 5, tel_te_pool_class, telugu_chars, "Handwritten Data")
vv = PCAretQ(tel_train_pool, 50)
tel_pca_acc, tel_pca_probs = LogisticRegression(tel_train_pool@vv, tel_tr_pool_class, tel_test_pool@vv, tel_te_pool_class,700, 0.001,5)
print("Logistic Regression: Handwritten Data(PCA) : Accuracy = "+str(tel_pca_acc))
confMatPlotter(tel_pca_probs, 5, tel_te_pool_class,telugu_chars, "Handwritten Data(PCA)")
vv = LDAMultiClass(5,tel_train, 50)
tel_lda_acc, tel_lda_probs = LogisticRegression(tel_train_pool@vv, tel_tr_pool_class, tel_test_pool@vv, tel_te_pool_class,700, 0.001,5)
print("Logistic Regression: Handwritten Data(LDA) : Accuracy = "+str(tel_lda_acc))
confMatPlotter(tel_lda_probs, 5, tel_te_pool_class, telugu_chars, "Handwritten Data(LDA)")


########### Image files

img_acc, img_probs = LogisticRegression(img_train_pool, img_train_class, img_test_pool, img_test_class,200, 0.0001,5)
print("Logistic Regression: Image Data : Accuracy = "+str(img_acc))
confMatPlotter(img_probs, 5, img_test_class, ['coast','forest','highway','moun','open'], "Image Data")
vv = PCAretQ(img_train_pool, 70)
img_pca_acc, img_pca_probs = LogisticRegression(img_train_pool@vv, img_train_class, img_test_pool@vv, img_test_class,200, 0.0001,5)
print("Logistic Regression: Image Data(PCA) : Accuracy = "+str(img_pca_acc))
confMatPlotter(img_pca_probs, 5, img_test_class,  ['coast','forest','highway','moun','open'], "Image Data(PCA)")
vv = LDAMultiClass(5,imgtrain828, 70)
img_lda_acc, img_lda_probs = LogisticRegression(img_train_pool@vv, img_train_class, img_test_pool@vv, img_test_class,200, 0.0001,5)
print("Logistic Regression: Image Data(LDA) : Accuracy = "+str(img_lda_acc))
confMatPlotter(img_lda_probs, 5, img_test_class,  ['coast','forest','highway','moun','open'], "Image Data(LDA)")

plot_ROC_DET(img_probs, img_pca_probs, img_lda_probs, 5, img_test_class, "Image Data")


fp = open("rocvals.txt",'w')
save_ROC(syn_probs, 2,syn_d_class, fp)
save_ROC(img_pca_probs, 5,img_test_class, fp)
save_ROC(aud_pca_probs, 5,aud_te_pool_class, fp)
save_ROC(tel_pca_probs, 5,tel_te_pool_class, fp)
fp.close()

"""fp = open("rocvals.txt",'r')
fpr_syn_lr = fp.readline().split()
tpr_syn_lr = fp.readline().split()
fnr_syn_lr = fp.readline().split()
fpr_img_lr = fp.readline().split()
tpr_img_lr = fp.readline().split()
fnr_img_lr = fp.readline().split()
fpr_aud_lr = fp.readline().split()
tpr_aud_lr = fp.readline().split()
fnr_aud_lr = fp.readline().split()
fpr_tel_lr = fp.readline().split()
tpr_tel_lr = fp.readline().split()
fnr_tel_lr = fp.readline().split()
fp.close()"""
