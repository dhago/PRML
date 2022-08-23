#!/usr/bin/env python
# coding: utf-8

# In[148]:


import sklearn
import numpy as np
from scipy.stats import norm
import copy
import os
from sklearn import svm
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from scipy import stats
import math
from sklearn.metrics import det_curve, DetCurveDisplay, confusion_matrix,  ConfusionMatrixDisplay


# In[99]:


#plotting the ROC

def plotROC(scores): 
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
        
#     display = sklearn.metrics.DetCurveDisplay(FPR, FNR, estimator_name=None, pos_label=None)
#     display.plot()
#     plt.show()
            
#     axis.plot(FPR,TPR)
#     axis.set_title(f"ROC Curve")
#     axis.set_xlabel("False Positive Rate(FPR)")
#     axis.set_ylabel("True Positive Rate(TPR)")
#     x=np.linspace(0,1,100)
#     y=np.linspace(0,1,100)
#     plt.plot(x,y)
#     plt.plot()
    
    return([FPR,TPR])


# In[3]:


def normalise(values):
    fmin = np.min(values,axis=0)
    fmax = np.max(values,axis=0)
    temp = (values - fmin)/(fmax-fmin)
    return temp


# In[4]:


def accuracy(output_acc):
    total = 0
    correct = 0
    for i in range(len(output_acc)):
        for j in range(len(output_acc)):
            total+=output_acc[i][j]
            if i==j:
                correct+=output_acc[i][j]
    return(correct/total)*100


# In[5]:


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


# In[6]:


def concatanater(array):
    c = 0
    for i in array:
        if c==0:
            array_concat = i
        else:
            array_concat = np.concatenate((array_concat,i))
        c+=1
    return array_concat


# In[7]:


def resizer(original, concat_arr, pca_or_not=0, num_eigen_taken=0):
    num_classes = len(original)
    sizes = [0]*num_classes
    for i in range(num_classes):
        sizes[i] = len(original[i])
    num_feat = len(original[0][0])
    
    ranges = np.zeros((num_classes,2))
    
    for i in range(num_classes):
        if i==0:
            t1 = 0
            t2 = sizes[i]-1
            ranges[i][0]=t1
            ranges[i][1]=t2
        else:
            t1=t2+1
            t2=t1-1+sizes[i]
            ranges[i][0]=t1
            ranges[i][1]=t2
            
    new_arr = []
    for i in sizes:
        if pca_or_not==0:
            temp = np.zeros((i,num_feat))
        else:
            temp = np.zeros((i,num_eigen_taken))
        new_arr.append(temp)
        
    for i in range(num_classes):
        #print(str(int(ranges[i][0]))+"   "+str(int(ranges[i][1])))
        new_arr[i] = concat_arr[int(ranges[i][0]):int(ranges[i][1]+1)]
        
    return(new_arr)


# In[8]:


#KNN

def KNN(all_train_data_array, test_point, k):
    
    num_classes=len(all_train_data_array)
    n=sum(len(i) for i in all_train_data_array)
    f=len(all_train_data_array[0][0])
    
    temp_train_data = copy.deepcopy(all_train_data_array)
    
    class_ = 0
    for i in temp_train_data:
        i = i-test_point #difference in points
        i = i*i #square of euclidean distance
        i = np.sum(i,axis=1)
        i = np.sqrt(i)
        temp = np.full(len(i),class_)
        i = np.c_[i,temp]
        
        if class_==0:
            temp_holder=i
        else:
            temp_holder = np.concatenate((temp_holder,i))
            
        class_ +=1
 
    temp_holder = temp_holder[np.argsort(temp_holder[:,0])] #sorts based on distances
    top_k_values = temp_holder[0:k]
    
    probs = [0]*num_classes
    dist = [0]*num_classes
    class_label = [0]*num_classes
    for i in top_k_values:
        probs[int(i[1])] += 1/k
        dist[int(i[1])] += i[0]
        
    for i in range(num_classes):
        dist[i] = dist[i]/probs[i]  if probs[i]!=0 else np.inf
    class_label=np.array(range(num_classes))
    
    temp = np.c_[probs,dist,class_label]
    
    temp = temp[np.argsort(temp[:,0])] #sorts based on probabilities then avg distances
    temp = temp[np.argsort(temp[:,1], kind='mergesort')]
    pred_class = int(temp[0][2])
    pred_probs = probs
    
    return([pred_class,pred_probs])


# In[9]:


def KNN_tester(dev_data, all_train_data_array, k):
    
    ROC_plot_inputs = []
    num_classes = len(all_train_data_array)
    act_vs_pred = np.zeros((num_classes,num_classes))
    
    actual_class = 0
    for i in dev_data:
        for j in i:
            temp_pred_class,temp_pred_probs = KNN(all_train_data_array, j, k)
            act_vs_pred[actual_class][temp_pred_class] += 1
            
            for z in range(num_classes):
                ROC_plot_inputs.append((temp_pred_probs[z],z,actual_class))
                
        actual_class += 1     
        
    return([act_vs_pred,ROC_plot_inputs])


# In[10]:


# a = np.array([[1,1],[2,3]])
# b = np.array([[1,2],[2,11],[5,5]])
# c = np.array([[1,2],[2,1],[5,5]])


# all_train_data_array = [a,b,c]
# k = 4
# KNN(all_train_data_array, [1.5,1.5], 6)


# In[11]:


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


# In[12]:


#PCA

def PCA(actual_data , threshold, thresh_or_num=0, num = 0):
    
    data = concatanater(actual_data)
    
    data_min_mean = data - np.mean(data , axis = 0)
    cov_mat = np.cov(data_min_mean , rowvar = False)
    
    eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)
    
    sorted_eigenvalue = eigen_values[np.argsort(np.absolute(eigen_values))[::-1]]
    sorted_eigenvectors = eigen_vectors[:,np.argsort(np.absolute(eigen_values))[::-1]]
    if thresh_or_num==0:
        num_eigen_values_taken = find_nearest(sorted_eigenvalue, threshold)+1
        eigenvector_subset = sorted_eigenvectors[:,0:num_eigen_values_taken]
    else:
        num_eigen_values_taken = num
        eigenvector_subset = sorted_eigenvectors[:,0:num_eigen_values_taken]  
     
    data_reduced = np.dot(eigenvector_subset.transpose() , data_min_mean.transpose() ).transpose()
    data_reduced_resized= resizer(actual_data, data_reduced, 1, num_eigen_values_taken)
     
    return data_reduced_resized,sorted_eigenvectors


# In[53]:


#LDA

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


# In[13]:
#now for svm:
def SVM_classifier(dev_data, all_train_data_array, kernel_given='linear', name =""):
    
    ROCinputs = []
    c = 0
    for i in all_train_data_array:
        if c==0:
            X_train = i
            y_train = np.full(len(i),c)
        else:
            X_train = np.concatenate((X_train,i))
            y_train = np.concatenate((y_train,np.full(len(i),c)))
        c+=1
    
    c = 0
    for i in dev_data:
        if c==0:
            X_test = i
            y_test = np.full(len(i),c)
        else:
            X_test = np.concatenate((X_test,i))
            y_test = np.concatenate((y_test,np.full(len(i),c)))
        c+=1
    
    
    clf = svm.SVC(kernel=kernel_given, probability = True, decision_function_shape="ovo", C = 10)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    probabs = clf.predict_proba(X_test)
    
    for i in range(len(probabs)):
        for j in range(len(all_train_data_array)):
            ROCinputs.append((probabs[i][j],j,y_test[i]))

    #evaluating the metric
    print("Accuracy:",100*metrics.accuracy_score(y_test, y_pred))
    conf = confusion_matrix(y_test, (y_pred),labels=[i for i in range(len(all_train_data_array))])
    display = ConfusionMatrixDisplay(confusion_matrix=conf, display_labels=[i for i in range(len(all_train_data_array))])
    display.plot()
    display.ax_.set_title('Confusion Matrix: for SVM '+kernel_given+" "+name, fontsize=10)
    plt.show()
    
    return(ROCinputs)

# In[26]:


#now for ANN:

def ANN_classifier(dev_data, all_train_data_array, hidden_layer_sizes, name=""):
    
    ROCinputs = []
    c = 0
    for i in all_train_data_array:
        if c==0:
            X_train = i
            y_train = np.full(len(i),c)
        else:
            X_train = np.concatenate((X_train,i))
            y_train = np.concatenate((y_train,np.full(len(i),c)))
        c+=1
    
    c = 0
    for i in dev_data:
        if c==0:
            X_test = i
            y_test = np.full(len(i),c)
        else:
            X_test = np.concatenate((X_test,i))
            y_test = np.concatenate((y_test,np.full(len(i),c)))
        c+=1
    
    
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation='relu', solver='adam', max_iter=1000)
    mlp.fit(X_train,y_train)

    y_pred = mlp.predict(X_test)
    probabs = mlp.predict_proba(X_test)
    
    for i in range(len(probabs)):
        for j in range(len(all_train_data_array)):
            ROCinputs.append((probabs[i][j],j,y_test[i]))

    #evaluating the metric
    num_classes = len(all_train_data_array)
    print("Accuracy:",100*metrics.accuracy_score(y_test, y_pred))
    conf = confusion_matrix(y_test, (y_pred),labels=[i for i in range(num_classes)])
    display = ConfusionMatrixDisplay(confusion_matrix=conf, display_labels=[i for i in range(num_classes)])
    display.plot()
    display.ax_.set_title('Confusion Matrix: ANN '+name, fontsize=15)
    plt.show()
    return ROCinputs


#getting the datas now:
#loading the synthetic train data:

file=open(r"19/train.txt","r")
c=0
c1c=0
c2c=0
traindata_syn_class1=np.zeros((1250,2))
traindata_syn_class2=np.zeros((1250,2))

for line in file:
    line=(list(map(float, line.strip().split(","))))
    c+=1
    if int(line[2])==1:
        traindata_syn_class1[c1c]=line[:2]
        c1c+=1
    else:
        traindata_syn_class2[c2c]=line[:2]
        c2c+=1
        
traindata_synthetic = [traindata_syn_class1,traindata_syn_class2]


# In[15]:


#loading the synthetic dev data:

file=open(r"19/dev.txt","r")
c=0
c1c=0
c2c=0
devdata_syn_class1=np.zeros((500,2))
devdata_syn_class2=np.zeros((500,2))

for line in file:
    line=(list(map(float, line.strip().split(","))))
    c+=1
    if int(line[2])==1:
        devdata_syn_class1[c1c]=line[:2]
        c1c+=1
    else:
        devdata_syn_class2[c2c]=line[:2]
        c2c+=1
        
devdata_synthetic = [devdata_syn_class1,devdata_syn_class2]
xx,syn_vv = PCA(traindata_synthetic, 1,1)
lda_syn_vv = LDAMultiClass(2, traindata_synthetic, 1)

# In[16]:
#################
pca_dev_syn = []
pca_train_syn = []
for i in range(2):
    pca_train_syn.append(traindata_synthetic[i]@syn_vv[:,0:1])
    pca_dev_syn.append(devdata_synthetic[i]@syn_vv[:,0:1])
lda_dev_syn = []
lda_train_syn = []
for i in range(2):
    lda_train_syn.append(traindata_synthetic[i]@lda_syn_vv[:,0:1])
    lda_dev_syn.append(devdata_synthetic[i]@lda_syn_vv[:,0:1])

k = 5
output_acc, ROC_inputs_syn = KNN_tester(devdata_synthetic, traindata_synthetic, k)
output_acc2, ROC_inputs_syn2 = KNN_tester(pca_dev_syn, pca_train_syn, k)
output_acc3, ROC_inputs_syn3 = KNN_tester(lda_dev_syn, lda_train_syn, k)
print("XXXXXXXXXXXXXXXXXXXXXX")
print("\n")
print("KNN: Synthetic Data : Accuracy: "+ str(accuracy(output_acc)))
print("KNN: Synthetic Data(PCA): Accuracy: "+ str(accuracy(output_acc2)))
print("KNN: Synthetic Data(LDA) : Accuracy: "+ str(accuracy(output_acc3)))

display = ConfusionMatrixDisplay(confusion_matrix=output_acc.astype(int), display_labels=[1,2])
display.plot()
display.ax_.set_title('Synthetic data Confusion Matrix for k ='+str(k), fontsize=15)
plt.show()
display = ConfusionMatrixDisplay(confusion_matrix=output_acc2.astype(int), display_labels=[1,2])
display.plot()
display.ax_.set_title('Synthetic data Confusion Matrix for PCA,k ='+str(k), fontsize=15)
plt.show()
display = ConfusionMatrixDisplay(confusion_matrix=output_acc3.astype(int), display_labels=[1,2])
display.plot()
display.ax_.set_title('Synthetic data Confusion Matrix for LDA,k ='+str(k), fontsize=15)
plt.show()

knn_syn_fpr1,knn_syn_tpr1= plotROC(ROC_inputs_syn)
knn_syn_fpr2,knn_syn_tpr2= plotROC(ROC_inputs_syn)
knn_syn_fpr3,knn_syn_tpr3= plotROC(ROC_inputs_syn)


##################
#Different kernels in SVM vs synthetic data

print("synthetic data, SVM with linear:")
l = SVM_classifier(devdata_synthetic, traindata_synthetic, "linear","Synthetic data")
w,u= plotROC(l)
print("\n")

print("synthetic data, SVM with rbf:")
l = SVM_classifier(devdata_synthetic, traindata_synthetic, "rbf","Synthetic data")
syn_svm_fpr1,syn_svm_tpr1= plotROC(l)
print("\n")

print("synthetic data, SVM with poly:")
l = SVM_classifier(devdata_synthetic, traindata_synthetic, "poly","Synthetic data")
w,u= plotROC(l)
print("\n")

print("synthetic data, SVM with sigmoid:")
l = SVM_classifier(devdata_synthetic, traindata_synthetic, "sigmoid","Synthetic data")
w,u= plotROC(l)

print("synthetic data(PCA), SVM with linear:")
l = SVM_classifier(pca_dev_syn, pca_train_syn, "linear","Synthetic data")
w,u= plotROC(l)
print("\n")

print("synthetic data(PCA), SVM with rbf:")
l = SVM_classifier(pca_dev_syn, pca_train_syn, "rbf","Synthetic data")
syn_svm_fpr2,syn_svm_tpr2 = plotROC(l)
print("\n")

print("synthetic data(PCA), SVM with poly:")
l = SVM_classifier(pca_dev_syn, pca_train_syn, "poly","Synthetic data")
w,u= plotROC(l)
print("\n")

print("synthetic data(PCA), SVM with sigmoid:")
l = SVM_classifier(pca_dev_syn, pca_train_syn, "sigmoid","Synthetic data")
w,u= plotROC(l)

print("synthetic data(LDA), SVM with linear:")
l = SVM_classifier(lda_dev_syn, lda_train_syn, "linear","Synthetic data")
w,u= plotROC(l)
print("\n")

print("synthetic data(LDA), SVM with rbf:")
l = SVM_classifier(lda_dev_syn, lda_train_syn, "rbf","Synthetic data")
syn_svm_fpr3,syn_svm_tpr3  = plotROC(l)
print("\n")

print("synthetic data(LDA), SVM with poly:")
l = SVM_classifier(lda_dev_syn, lda_train_syn, "poly","Synthetic data")
w,u= plotROC(l)
print("\n")

print("synthetic data(LDA), SVM with sigmoid:")
l = SVM_classifier(lda_dev_syn, lda_train_syn, "sigmoid","Synthetic data")
w,u= plotROC(l)

fig, (ax_roc,ax_det) = plt.subplots(1,2,figsize=(20, 10))
fig.suptitle('Synthetic data: ROC and DET for SVM(rbf)', fontsize=20)
ax_roc.plot([0,1],[0,1], linestyle="--", color='black')
l1=ax_roc.plot(syn_svm_fpr1,syn_svm_tpr1,label='Without PCA/LDA')
l2=ax_roc.plot(syn_svm_fpr2,syn_svm_tpr2,label='After PCA')
l3=ax_roc.plot(syn_svm_fpr3,syn_svm_tpr3,label='After LDA')
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_title('ROC curves', fontsize=15)
ax_roc.legend(loc=0, prop={'size': 20})
DetCurveDisplay(fpr=syn_svm_fpr1, fnr=[1-i for i in syn_svm_tpr1], estimator_name='Without PCA/LDA').plot(ax = ax_det)
DetCurveDisplay(fpr=syn_svm_fpr2, fnr=[1-i for i in syn_svm_tpr2], estimator_name='After PCA').plot(ax = ax_det)
DetCurveDisplay(fpr=syn_svm_fpr3, fnr=[1-i for i in syn_svm_tpr3], estimator_name='After LDA').plot(ax = ax_det)
ax_det.set_title('DET curves', fontsize=15)
ax_det.legend(loc=0, prop={'size': 20})

#############

# In[27]:
print("Synthetic data, ANN with parameters = : (40,20,20,20)")
l1 = ANN_classifier(devdata_synthetic, traindata_synthetic, (40,20),"") #(20,20,20,20),(128,64,32,32) works well - 73%
print("Synthetic data(PCA), ANN with parameters = : (40,20,20,20)")
l2 = ANN_classifier(pca_dev_syn, pca_train_syn, (40,20), "(PCA)") #(20,20,20,20),(128,64,32,32) works well - 73%
print("Synthetic data(LDA), ANN with parameters = : (40,20,20,20)")
l3 = ANN_classifier(lda_dev_syn, lda_train_syn, (40,20), "(LDA)") #(20,20,20,20),(128,64,32,32) works well - 73%
ann_syn_fpr1,ann_syn_tpr1= plotROC(l1)
ann_syn_fpr2,ann_syn_tpr2= plotROC(l2)
ann_syn_fpr3,ann_syn_tpr3= plotROC(l3)


fig, (ax_roc,ax_det) = plt.subplots(1,2,figsize=(20, 10))
fig.suptitle('Synthetic data: ROC and DET for ANN', fontsize=20)
ax_roc.plot([0,1],[0,1], linestyle="--", color='black')
l1=ax_roc.plot(ann_syn_fpr1,ann_syn_tpr1,label='Without PCA/LDA')
l2=ax_roc.plot(ann_syn_fpr2,ann_syn_tpr2,label='After PCA')
l3=ax_roc.plot(ann_syn_fpr3,ann_syn_tpr3,label='After LDA')
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_title('ROC curves', fontsize=15)
ax_roc.legend(loc=0, prop={'size': 20})
DetCurveDisplay(fpr=ann_syn_fpr1, fnr=[1-i for i in ann_syn_tpr1], estimator_name='Without PCA/LDA').plot(ax = ax_det)
DetCurveDisplay(fpr=ann_syn_fpr2, fnr=[1-i for i in ann_syn_tpr2], estimator_name='After PCA').plot(ax = ax_det)
DetCurveDisplay(fpr=ann_syn_fpr3, fnr=[1-i for i in ann_syn_tpr3], estimator_name='After LDA').plot(ax = ax_det)
ax_det.set_title('DET curves', fontsize=15)
ax_det.legend(loc=0, prop={'size': 20})

# In[17]:
#For Image data:

#function that loads the data but 36*23=828
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


# In[18]:

base_path = os.getcwd()
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



# In[24]:


scaler = StandardScaler()
    
imgtrain828_concat = concatanater(imgtrain828)
imgdev828_concat = concatanater(imgdev828)

scaler.fit(imgtrain828_concat)  
imgtrain828_normalised = scaler.transform(imgtrain828_concat)  
imgdev828_normalised = scaler.transform(imgdev828_concat)  

imgtrain828_norm_resized = resizer(imgtrain828, imgtrain828_normalised)
imgdev828_norm_resized = resizer(imgdev828, imgdev828_normalised)


# In[25]:
xx,syn_vv = PCA(imgtrain828_norm_resized, 1,100)
lda_syn_vv = LDAMultiClass(5, imgtrain828_norm_resized, 200)

# In[16]:

pca_dev_img = []
pca_train_img = []
for i in range(5):
    pca_train_img.append(imgtrain828_norm_resized[i]@syn_vv[:,0:70])
    pca_dev_img.append(imgdev828_norm_resized[i]@syn_vv[:,0:70])
lda_dev_img = []
lda_train_img = []
for i in range(5):
    lda_train_img.append(imgtrain828_norm_resized[i]@lda_syn_vv[:,0:70])
    lda_dev_img.append(imgdev828_norm_resized[i]@lda_syn_vv[:,0:70])

k = 3
output_acc, ROC_inputs_img1 = KNN_tester(imgdev828_norm_resized, imgtrain828_norm_resized, k)
output_acc2, ROC_inputs_img2 = KNN_tester(pca_dev_img, pca_train_img, k)
output_acc3, ROC_inputs_img3 = KNN_tester(lda_dev_img, lda_train_img, k)
print("XXXXXXXXXXXXXXXXXXXXXX")
print("\n")
print("Image DATA")
print("accuracy achieved using k="+str(k)+" and normalising is : "+str(accuracy(output_acc)))
print("accuracy achieved using PCA,k="+str(k)+" and normalising is : "+str(accuracy(output_acc2)))
print("accuracy achieved using LDA,k="+str(k)+" and normalising is : "+str(accuracy(output_acc3)))

display = ConfusionMatrixDisplay(confusion_matrix=output_acc, display_labels=[0,1,2,3,4])
display.plot()
display.ax_.set_title('Image data: Confusion Matrix for k ='+str(k), fontsize=15)
plt.show()
display = ConfusionMatrixDisplay(confusion_matrix=output_acc2, display_labels=[0,1,2,3,4])
display.plot()
display.ax_.set_title('Image data: Confusion Matrix for PCA,k ='+str(k), fontsize=15)
plt.show()
display = ConfusionMatrixDisplay(confusion_matrix=output_acc3, display_labels=[0,1,2,3,4])
display.plot()
display.ax_.set_title('Image data: Confusion Matrix for LDA,k ='+str(k), fontsize=15)
plt.show()
knn_img_fpr1,knn_img_tpr1= plotROC(ROC_inputs_img1)
knn_img_fpr2,knn_img_tpr2= plotROC(ROC_inputs_img2)
knn_img_fpr3,knn_img_tpr3= plotROC(ROC_inputs_img3)



#SVMs
print("image data, SVM with linear:")
l = SVM_classifier(imgdev828_norm_resized, imgtrain828_norm_resized, "linear","Image data")
#plotROC(l)
print("\n")

print("image data, SVM with rbf:")
l1 = SVM_classifier(imgdev828_norm_resized, imgtrain828_norm_resized, "rbf","Image data")
img_svm_fpr1,img_svm_tpr1 = plotROC(l1)
print("\n")

print("image data, SVM with poly:")
l = SVM_classifier(imgdev828_norm_resized, imgtrain828_norm_resized, "poly","Image data")
#plotROC(l)
print("\n")

print("image data, SVM with sigmoid:")
l = SVM_classifier(imgdev828_norm_resized, imgtrain828_norm_resized, "sigmoid","Image data")
#plotROC(l)
print("\n")

print("image data(PCA), SVM with linear:")
l = SVM_classifier(pca_dev_img, pca_train_img, "linear","Image data(PCA)")
#plotROC(l)
print("\n")

print("image data(PCA), SVM with rbf:")
l2 = SVM_classifier(pca_dev_img, pca_train_img, "rbf","Image data(PCA)")
img_svm_fpr2,img_svm_tpr2 = plotROC(l2)
print("\n")

print("image data(PCA), SVM with poly:")
l = SVM_classifier(pca_dev_img, pca_train_img, "poly","Image data(PCA)")
#plotROC(l)
print("\n")

print("image data(PCA), SVM with sigmoid:")
l = SVM_classifier(pca_dev_img, pca_train_img, "sigmoid","Image data(PCA)")
#plotROC(l)
print("\n")

print("image data(LDA), SVM with linear:")
l = SVM_classifier(lda_dev_img, lda_train_img, "linear","Image data(LDA)")
#plotROC(l)
print("\n")

print("image data(LDA), SVM with rbf:")
l3 = SVM_classifier(lda_dev_img, lda_train_img, "rbf","Image data(LDA)")
img_svm_fpr3,img_svm_tpr3 = plotROC(l3)
print("\n")

print("image data(LDA), SVM with poly:")
l = SVM_classifier(lda_dev_img, lda_train_img, "poly","Image data(LDA)")
#plotROC(l)
print("\n")

print("image data(LDA), SVM with sigmoid:")
l = SVM_classifier(lda_dev_img, lda_train_img, "sigmoid","Image data(LDA)")
#plotROC(l)
print("\n")


svm_img_fpr1,svm_img_tpr1= plotROC(l1)
svm_img_fpr2,svm_img_tpr2= plotROC(l2)
svm_img_fpr3,svm_img_tpr3= plotROC(l3)


fig, (ax_roc,ax_det) = plt.subplots(1,2,figsize=(20, 10))
fig.suptitle('Inage data: ROC and DET for SVM', fontsize=20)
ax_roc.plot([0,1],[0,1], linestyle="--", color='black')
l1=ax_roc.plot(svm_img_fpr1,svm_img_tpr1,label='Without PCA/LDA')
l2=ax_roc.plot(svm_img_fpr2,svm_img_tpr2,label='After PCA')
l3=ax_roc.plot(svm_img_fpr3,svm_img_tpr3,label='After LDA')
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_title('ROC curves', fontsize=15)
ax_roc.legend(loc=0, prop={'size': 20})
DetCurveDisplay(fpr=svm_img_fpr1, fnr=[1-i for i in svm_img_tpr1], estimator_name='Without PCA/LDA').plot(ax = ax_det)
DetCurveDisplay(fpr=svm_img_fpr2, fnr=[1-i for i in svm_img_tpr2], estimator_name='After PCA').plot(ax = ax_det)
DetCurveDisplay(fpr=svm_img_fpr3, fnr=[1-i for i in svm_img_tpr3], estimator_name='After LDA').plot(ax = ax_det)
ax_det.set_title('DET curves', fontsize=15)
ax_det.legend(loc=0, prop={'size': 20})


print("image data, ANN :")
l1 = ANN_classifier(imgdev828_norm_resized, imgtrain828_norm_resized, (100,10),"Images") #(20,20,20,20),(128,64,32,32) works well - 73%
print("image data(PCA), ANN :")
l2 = ANN_classifier(pca_dev_img, pca_train_img, (100,10), "Images (PCA)") #(20,20,20,20),(128,64,32,32) works well - 73%
print("image data(LDA), ANN :")
l3 = ANN_classifier(lda_dev_img, lda_train_img, (100,10), "Images (LDA)") #(20,20,20,20),(128,64,32,32) works well - 73%
ann_img_fpr1,ann_img_tpr1= plotROC(l1)
ann_img_fpr2,ann_img_tpr2= plotROC(l2)
ann_img_fpr3,ann_img_tpr3= plotROC(l3)


fig, (ax_roc,ax_det) = plt.subplots(1,2,figsize=(20, 10))
fig.suptitle('Image data : ROC and DET for ANN', fontsize=20)
ax_roc.plot([0,1],[0,1], linestyle="--", color='black')
l1=ax_roc.plot(ann_img_fpr1,ann_img_tpr1,label='Without PCA/LDA')
l2=ax_roc.plot(ann_img_fpr2,ann_img_tpr2,label='After PCA')
l3=ax_roc.plot(ann_img_fpr3,ann_img_tpr3,label='After LDA')
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_title('ROC curves', fontsize=15)
ax_roc.legend(loc=0, prop={'size': 20})
DetCurveDisplay(fpr=ann_img_fpr1, fnr=[1-i for i in ann_img_tpr1], estimator_name='Without PCA/LDA').plot(ax = ax_det)
DetCurveDisplay(fpr=ann_img_fpr2, fnr=[1-i for i in ann_img_tpr2], estimator_name='After PCA').plot(ax = ax_det)
DetCurveDisplay(fpr=ann_img_fpr3, fnr=[1-i for i in ann_img_tpr3], estimator_name='After LDA').plot(ax = ax_det)
ax_det.set_title('DET curves', fontsize=15)
ax_det.legend(loc=0, prop={'size': 20})




# In[32]:tel_test
def z_score_normalisation(indices, data_from_files):
    temp = {}
    for i in indices:
        temp[i] = {}
        for file in data_from_files[i]:
            temp[i][file] = stats.zscore(data_from_files[i][file])
    return temp


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
test_audio_data = z_score_normalisation(inputChars, test_audio_data)
train_audio_data = z_score_normalisation(inputChars, train_audio_data)
# In[37]:


def readTeluguChars (filepath):
    with open(filepath, 'r') as fp:
        a = fp.readline().strip().split(" ") 
        size = int(a[0])
    return size,np.array(a[1:], 'float64').reshape([size,2])


# In[38]:


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
    


#temp_tel_train = min_max_normalisation(telugu_chars, tel_train_data)
tel_test = z_score_normalisation(telugu_chars, tel_test)
tel_train = z_score_normalisation(telugu_chars, tel_train)
def zscore_normalisation_v2(data):
    return stats.zscore(data)
def min_max_normalisation_v2(data):
    fmin = np.min(data,axis=0)
    fmax = np.max(data,axis=0)
    return (data - fmin)/(fmax-fmin)


# In[39]:


def ChangeLengthAndFlatten(train,train_nv,test,test_nv,chars):
    final_train = []
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
        for file in train[char]:
            window = train_nv[char][file]-min_len+1
            x = []
            for i in range(min_len):
                avg = np.mean(train[char][file][i:i+window],axis=0).tolist()
                x.extend(avg)
            temp1.append(x)
        for file in test[char]:
            window = test_nv[char][file]-min_len+1
            x = []
            for i in range(min_len):
                avg = np.mean(test[char][file][i:i+window],axis=0).tolist()
                x.extend(avg)
            temp2.append(x)
        final_train.append(np.array(temp1))
        final_test.append(np.array(temp2))
    return final_train,final_test,min_len


# In[42]:


flattened_train_audio, flattened_test_audio,_ = ChangeLengthAndFlatten(train_audio_data, nv_train, test_audio_data, nv_test, inputChars)
flattened_train_handwritten,flattened_test_handwritten,_=ChangeLengthAndFlatten(tel_train, tel_train_sizes, tel_test, tel_test_sizes, telugu_chars)


# In[43]:

xx,syn_vv = PCA(flattened_train_audio, 1,100)
lda_syn_vv = LDAMultiClass(5, flattened_train_audio, 200)
#KNN for audio
pca_dev_aud = []
pca_train_aud = []
for i in range(5):
    pca_train_aud.append(flattened_train_audio[i]@syn_vv[:,0:50])
    pca_dev_aud.append(flattened_test_audio[i]@syn_vv[:,0:50])
lda_dev_aud = []
lda_train_aud = []
for i in range(5):
    lda_train_aud.append(flattened_train_audio[i]@lda_syn_vv[:,0:50])
    lda_dev_aud.append(flattened_test_audio[i]@lda_syn_vv[:,0:50])


k = 3
output_acc, ROC_inputs_aud1 = KNN_tester(flattened_test_audio, flattened_train_audio, k)
output_acc2, ROC_inputs_aud2 = KNN_tester(pca_dev_aud, pca_train_aud, k)
output_acc3, ROC_inputs_aud3 = KNN_tester(lda_dev_aud, lda_train_aud, k)
print("XXXXXXXXXXXXXXXXXXXXXX")
print("\n")
print("For spoken digit dataset: ")
print("accuracy achieved using k="+str(k)+" and normalising is : "+str(accuracy(output_acc)))
print("accuracy achieved using PCA,k="+str(k)+" and normalising is : "+str(accuracy(output_acc2)))
print("accuracy achieved using LDA,k="+str(k)+" and normalising is : "+str(accuracy(output_acc3)))

display = ConfusionMatrixDisplay(confusion_matrix=output_acc, display_labels=[1,2,3,4,9])
display.plot()
display.ax_.set_title('Audio data: Confusion Matrix for k ='+str(k), fontsize=10)
plt.show()
display = ConfusionMatrixDisplay(confusion_matrix=output_acc2, display_labels=[1,2,3,4,9])
display.plot()
display.ax_.set_title('Audio data: Confusion Matrix for PCA,k ='+str(k), fontsize=10)
plt.show()
display = ConfusionMatrixDisplay(confusion_matrix=output_acc3, display_labels=[1,2,3,4,9])
display.plot()
display.ax_.set_title('Audio data: Confusion Matrix for LDA,k ='+str(k), fontsize=10)
plt.show()
knn_aud_fpr1,knn_aud_tpr1= plotROC(ROC_inputs_aud1)
knn_aud_fpr2,knn_aud_tpr2= plotROC(ROC_inputs_aud2)
knn_aud_fpr3,knn_aud_tpr3= plotROC(ROC_inputs_aud3)

# In[44]:


#SVM for audio

print("audio data, SVM with linear:")
l = SVM_classifier(flattened_test_audio, flattened_train_audio, "linear","Audio data")
print("\n")

print("audio data, SVM with rbf:")
l = SVM_classifier(flattened_test_audio, flattened_train_audio, "rbf","Audio data")
print("\n")

print("audio data, SVM with poly:")
l = SVM_classifier(flattened_test_audio, flattened_train_audio, "poly","Audio data")
print("\n")

print("audio data, SVM with sigmoid:")
l1 = SVM_classifier(flattened_test_audio, flattened_train_audio, "sigmoid","Audio data")
print("\n")

print("audio data(PCA), SVM with linear:")
l = SVM_classifier(pca_dev_aud, pca_train_aud, "linear","Audio data(PCA)")
print("\n")

print("audio data(PCA), SVM with rbf:")
l = SVM_classifier(pca_dev_aud, pca_train_aud, "rbf","Audio data(PCA)")
print("\n")

print("audio data(PCA), SVM with poly:")
l = SVM_classifier(pca_dev_aud, pca_train_aud, "poly","Audio data(PCA)")
print("\n")

print("audio data(PCA), SVM with sigmoid:")
l2 = SVM_classifier(pca_dev_aud, pca_train_aud, "sigmoid","Audio data(PCA)")
print("\n")

print("audio data(LDA), SVM with linear:")
l = SVM_classifier(lda_dev_aud, lda_train_aud, "linear","Audio data(LDA)")
print("\n")

print("audio data(LDA), SVM with rbf:")
l = SVM_classifier(lda_dev_aud, lda_train_aud, "rbf","Audio data(LDA)")
print("\n")

print("audio data(LDA), SVM with poly:")
l = SVM_classifier(lda_dev_aud, lda_train_aud, "poly","Audio data(LDA)")
print("\n")

print("audio data(LDA), SVM with sigmoid:")
l3 = SVM_classifier(lda_dev_aud, lda_train_aud, "sigmoid","Audio data(LDA)")
print("\n")


svm_aud_fpr1,svm_aud_tpr1= plotROC(l1)
svm_aud_fpr2,svm_aud_tpr2= plotROC(l2)
svm_aud_fpr3,svm_aud_tpr3= plotROC(l3)


fig, (ax_roc,ax_det) = plt.subplots(1,2,figsize=(20, 10))
fig.suptitle('Audio data: ROC and DET for SVM(sigmoid)', fontsize=20)
ax_roc.plot([0,1],[0,1], linestyle="--", color='black')
l1=ax_roc.plot(svm_aud_fpr1,svm_aud_tpr1,label='Without PCA/LDA')
l2=ax_roc.plot(svm_aud_fpr2,svm_aud_tpr2,label='After PCA')
l3=ax_roc.plot(svm_aud_fpr3,svm_aud_tpr3,label='After LDA')
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_title('ROC curves', fontsize=15)
ax_roc.legend(loc=0, prop={'size': 20})
DetCurveDisplay(fpr=svm_aud_fpr1, fnr=[1-i for i in svm_aud_tpr1], estimator_name='Without PCA/LDA').plot(ax = ax_det)
DetCurveDisplay(fpr=svm_aud_fpr2, fnr=[1-i for i in svm_aud_tpr2], estimator_name='After PCA').plot(ax = ax_det)
DetCurveDisplay(fpr=svm_aud_fpr3, fnr=[1-i for i in svm_aud_tpr3], estimator_name='After LDA').plot(ax = ax_det)
ax_det.set_title('DET curves', fontsize=15)
ax_det.legend(loc=0, prop={'size': 20})

# In[48]:


#ANN on audio

print("audio data, ANN with parameters = : (128,32,32,32)")
l1 = ANN_classifier(flattened_test_audio, flattened_train_audio, (128,32), "") #(20,20,20,20) works well - 73%
print("audio data(PCA), ANN with parameters = :(128,32,32,32)")
l2 = ANN_classifier(pca_dev_aud, pca_train_aud, (128,32), "(PCA)")  #(20,20,20,20) works well - 73%
print("audio data(LDA), ANN with parameters = :(128,32,32,32)")
l3 = ANN_classifier(lda_dev_aud, lda_train_aud, (128,32), "(LDA)")  #(20,20,20,20) works well - 73%
ann_aud_fpr1,ann_aud_tpr1= plotROC(l1)
ann_aud_fpr2,ann_aud_tpr2= plotROC(l2)
ann_aud_fpr3,ann_aud_tpr3= plotROC(l3)


fig, (ax_roc,ax_det) = plt.subplots(1,2,figsize=(20, 10))
fig.suptitle('Audio data: ROC and DET for ANN', fontsize=20)
ax_roc.plot([0,1],[0,1], linestyle="--", color='black')
l1=ax_roc.plot(ann_aud_fpr1,ann_aud_tpr1,label='Without PCA/LDA')
l2=ax_roc.plot(ann_aud_fpr2,ann_aud_tpr2,label='After PCA')
l3=ax_roc.plot(ann_aud_fpr3,ann_aud_tpr3,label='After LDA')
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_title('ROC curves', fontsize=15)
ax_roc.legend(loc=0, prop={'size': 20})
DetCurveDisplay(fpr=ann_aud_fpr1, fnr=[1-i for i in ann_aud_tpr1], estimator_name='Without PCA/LDA').plot(ax = ax_det)
DetCurveDisplay(fpr=ann_aud_fpr2, fnr=[1-i for i in ann_aud_tpr2], estimator_name='After PCA').plot(ax = ax_det)
DetCurveDisplay(fpr=ann_aud_fpr3, fnr=[1-i for i in ann_aud_tpr3], estimator_name='After LDA').plot(ax = ax_det)
ax_det.set_title('DET curves', fontsize=15)
ax_det.legend(loc=0, prop={'size': 20})

# In[49]:


#KNN on handwritten


xx,syn_vv = PCA(flattened_train_handwritten, 1,100)
lda_syn_vv = LDAMultiClass(5, flattened_train_handwritten, 200)
#KNN for telio
pca_dev_tel = []
pca_train_tel = []
for i in range(5):
    pca_train_tel.append(flattened_train_handwritten[i]@syn_vv[:,0:50])
    pca_dev_tel.append(flattened_test_handwritten[i]@syn_vv[:,0:50])
lda_dev_tel = []
lda_train_tel = []
for i in range(5):
    lda_train_tel.append(flattened_train_handwritten[i]@lda_syn_vv[:,0:50])
    lda_dev_tel.append(flattened_test_handwritten[i]@lda_syn_vv[:,0:50])


k = 3
output_acc, ROC_inputs_tel1 = KNN_tester(flattened_test_handwritten, flattened_train_handwritten, k)
output_acc2, ROC_inputs_tel2 = KNN_tester(pca_dev_tel, pca_train_tel, k)
output_acc3, ROC_inputs_tel3 = KNN_tester(lda_dev_tel, lda_train_tel, k)
print("XXXXXXXXXXXXXXXXXXXXXX")
print("\n")
print("For handwritten dataset: ")
print("accuracy achieved using k="+str(k)+" and normalising is : "+str(accuracy(output_acc)))
print("accuracy achieved using PCA,k="+str(k)+" and normalising is : "+str(accuracy(output_acc2)))
print("accuracy achieved using LDA,k="+str(k)+" and normalising is : "+str(accuracy(output_acc3)))

display = ConfusionMatrixDisplay(confusion_matrix=output_acc, display_labels=telugu_chars)
display.plot()
display.ax_.set_title('Handwritten data : Confusion Matrix for k ='+str(k), fontsize=10)
plt.show()
display = ConfusionMatrixDisplay(confusion_matrix=output_acc2, display_labels=telugu_chars)
display.plot()
display.ax_.set_title('Handwritten data : Confusion Matrix for PCA,k ='+str(k), fontsize=10)
plt.show()
display = ConfusionMatrixDisplay(confusion_matrix=output_acc3, display_labels=telugu_chars)
display.plot()
display.ax_.set_title('Handwritten data : Confusion Matrix for LDA,k ='+str(k), fontsize=10)
plt.show()
knn_tel_fpr1,knn_tel_tpr1= plotROC(ROC_inputs_tel1)
knn_tel_fpr2,knn_tel_tpr2= plotROC(ROC_inputs_tel2)
knn_tel_fpr3,knn_tel_tpr3= plotROC(ROC_inputs_tel3)

# In[50]:


#SVM on handwritten

print("handwritten data, SVM with linear:")
l = SVM_classifier(flattened_test_handwritten, flattened_train_handwritten, "linear" ,"Handwritten data" )
print("\n")

print("handwritten data, SVM with rbf:")
l1 = SVM_classifier(flattened_test_handwritten, flattened_train_handwritten, "rbf","Handwritten data" )
print("\n")

print("handwritten data, SVM with poly:")
l = SVM_classifier(flattened_test_handwritten, flattened_train_handwritten, "poly","Handwritten data" )
print("\n")

print("handwritten data, SVM with sigmoid:")
l = SVM_classifier(flattened_test_handwritten, flattened_train_handwritten, "sigmoid","Handwritten data" )
print("\n")
print("handwritten data(PCA), SVM with linear:")
l = SVM_classifier(pca_dev_tel, pca_train_tel, "linear")
print("\n")

print("handwritten data(PCA), SVM with rbf:")
l2 = SVM_classifier(pca_dev_tel, pca_train_tel, "rbf","Handwritten data" )
print("\n")

print("handwritten data(PCA), SVM with poly:")
l = SVM_classifier(pca_dev_tel, pca_train_tel, "poly","Handwritten data" )
print("\n")

print("handwritten data(PCA), SVM with sigmoid:")
l = SVM_classifier(pca_dev_tel, pca_train_tel, "sigmoid","Handwritten data" )
print("\n")
print("handwritten data(LDA), SVM with linear:")
l = SVM_classifier(lda_dev_tel, lda_train_tel, "linear","Handwritten data" )
print("\n")

print("handwritten data(LDA), SVM with rbf:")
l3 = SVM_classifier(lda_dev_tel, lda_train_tel, "rbf","Handwritten data" )
print("\n")

print("handwritten data(LDA), SVM with poly:")
l = SVM_classifier(lda_dev_tel, lda_train_tel, "poly","Handwritten data" )
print("\n")

print("handwritten data(LDA), SVM with sigmoid:")
l = SVM_classifier(lda_dev_tel, lda_train_tel, "sigmoid","Handwritten data" )
print("\n")

svm_tel_fpr1,svm_tel_tpr1= plotROC(l1)
svm_tel_fpr2,svm_tel_tpr2= plotROC(l2)
svm_tel_fpr3,svm_tel_tpr3= plotROC(l3)


fig, (ax_roc,ax_det) = plt.subplots(1,2,figsize=(20, 10))
fig.suptitle('Handwritten data: ROC and DET for SVM', fontsize=20)
ax_roc.plot([0,1],[0,1], linestyle="--", color='black')
l1=ax_roc.plot(svm_tel_fpr1,svm_tel_tpr1,label='Without PCA/LDA')
l2=ax_roc.plot(svm_tel_fpr2,svm_tel_tpr2,label='After PCA')
l3=ax_roc.plot(svm_tel_fpr3,svm_tel_tpr3,label='After LDA')
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_title('ROC curves', fontsize=15)
ax_roc.legend(loc=0, prop={'size': 20})
DetCurveDisplay(fpr=svm_tel_fpr1, fnr=[1-i for i in svm_tel_tpr1], estimator_name='Without PCA/LDA').plot(ax = ax_det)
DetCurveDisplay(fpr=svm_tel_fpr2, fnr=[1-i for i in svm_tel_tpr2], estimator_name='After PCA').plot(ax = ax_det)
DetCurveDisplay(fpr=svm_tel_fpr3, fnr=[1-i for i in svm_tel_tpr3], estimator_name='After LDA').plot(ax = ax_det)
ax_det.set_title('DET curves', fontsize=15)
ax_det.legend(loc=0, prop={'size': 20})

# In[52]:


#ANN on handwritten

print("Handwritten data, ANN with parameters = : (128,64,32,32)")

l1 = ANN_classifier(flattened_test_handwritten, flattened_train_handwritten, (128,32,32,32), "") #(20,20,20,20) works well - 73%
print("Handwritten data(PCA), ANN with parameters = :(128,64,32,32)")
l2 = ANN_classifier(pca_dev_tel, pca_train_tel, (128,32,32,32), "(PCA)")  #(20,20,20,20) works well - 73%
print("Handwritten data(LDA), ANN with parameters = :(128,64,32,32)")
l3 = ANN_classifier(lda_dev_tel, lda_train_tel, (128,32,32,32), "(LDA)")  #(20,20,20,20) works well - 73%
ann_tel_fpr1,ann_tel_tpr1= plotROC(l1)
ann_tel_fpr2,ann_tel_tpr2= plotROC(l2)
ann_tel_fpr3,ann_tel_tpr3= plotROC(l3)


fig, (ax_roc,ax_det) = plt.subplots(1,2,figsize=(20, 10))
fig.suptitle('Handwritten data: ROC and DET for ANN', fontsize=20)
ax_roc.plot([0,1],[0,1], linestyle="--", color='black')
l1=ax_roc.plot(ann_tel_fpr1,ann_tel_tpr1,label='Without PCA/LDA')
l2=ax_roc.plot(ann_tel_fpr2,ann_tel_tpr2,label='After PCA')
l3=ax_roc.plot(ann_tel_fpr3,ann_tel_tpr3,label='After LDA')
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_title('ROC curves', fontsize=15)
ax_roc.legend(loc=0, prop={'size': 20})
DetCurveDisplay(fpr=ann_tel_fpr1, fnr=[1-i for i in ann_tel_tpr1], estimator_name='Without PCA/LDA').plot(ax = ax_det)
DetCurveDisplay(fpr=ann_tel_fpr2, fnr=[1-i for i in ann_tel_tpr2], estimator_name='After PCA').plot(ax = ax_det)
DetCurveDisplay(fpr=ann_tel_fpr3, fnr=[1-i for i in ann_tel_tpr3], estimator_name='After LDA').plot(ax = ax_det)
ax_det.set_title('DET curves', fontsize=15)
ax_det.legend(loc=0, prop={'size': 20})





fp = open("rocvals.txt",'r')
tpr_syn_lr = [float(i) for i in fp.readline().split()]
fpr_syn_lr = [float(i) for i in fp.readline().split()]
tpr_img_lr = [float(i) for i in fp.readline().split()]
fpr_img_lr = [float(i) for i in fp.readline().split()]
tpr_aud_lr = [float(i) for i in fp.readline().split()]
fpr_aud_lr = [float(i) for i in fp.readline().split()]
tpr_tel_lr = [float(i) for i in fp.readline().split()]
fpr_tel_lr = [float(i) for i in fp.readline().split()]
fp.close()

fig, (ax_roc,ax_det) = plt.subplots(1,2,figsize=(20, 10))
fig.suptitle('Comparing models: ROC and DET for Synthetic Data', fontsize=20)
ax_roc.plot([0,1],[0,1], linestyle="--", color='black')
l2=ax_roc.plot(fpr_syn_lr,tpr_syn_lr,label='Logistic Regression :')
l3=ax_roc.plot(syn_svm_fpr1,syn_svm_tpr1,label='SVM(rbf)')
l3=ax_roc.plot(ann_syn_fpr1,ann_syn_tpr1,label='ANN: Config: (40,20,20,20)')
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_title('ROC curves', fontsize=15)
ax_roc.legend(loc=0, prop={'size': 20})
DetCurveDisplay(fpr=fpr_syn_lr, fnr=[1-i for i in tpr_syn_lr], estimator_name='Logistic Regression :').plot(ax = ax_det)
DetCurveDisplay(fpr=syn_svm_fpr1, fnr=[1-i for i in syn_svm_tpr1], estimator_name='SVM(rbf)').plot(ax = ax_det)
DetCurveDisplay(fpr=ann_syn_fpr1, fnr=[1-i for i in ann_syn_tpr1], estimator_name='ANN: Config: (40,20,20,20)').plot(ax = ax_det)
ax_det.set_title('DET curves', fontsize=15)
ax_det.legend(loc=0, prop={'size': 20})

fig, (ax_roc,ax_det) = plt.subplots(1,2,figsize=(20, 10))
fig.suptitle('Comparing models: ROC and DET for Image Data', fontsize=20)
ax_roc.plot([0,1],[0,1], linestyle="--", color='black')
l2=ax_roc.plot(fpr_img_lr,tpr_img_lr,label='Logistic Regression (PCA) :')
l3=ax_roc.plot(img_svm_fpr2,img_svm_tpr2,label='SVM(rbf) with PCA' )
l3=ax_roc.plot(ann_img_fpr1,ann_img_tpr1,label='ANN: Config: (100,10)')
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_title('ROC curves', fontsize=15)
ax_roc.legend(loc=0, prop={'size': 20})
DetCurveDisplay(fpr=fpr_img_lr, fnr=[1-i for i in tpr_img_lr], estimator_name='Logistic Regression (PCA):').plot(ax = ax_det)
DetCurveDisplay(fpr=img_svm_fpr2, fnr=[1-i for i in img_svm_tpr2], estimator_name='SVM(rbf) with PCA').plot(ax = ax_det)
DetCurveDisplay(fpr=ann_img_fpr1, fnr=[1-i for i in ann_img_tpr1], estimator_name='ANN: Config: (100,10)').plot(ax = ax_det)
ax_det.set_title('DET curves', fontsize=15)
ax_det.legend(loc=0, prop={'size': 20})

fig, (ax_roc,ax_det) = plt.subplots(1,2,figsize=(20, 10))
fig.suptitle('Comparing models: ROC and DET for Audio Data', fontsize=20)
ax_roc.plot([0,1],[0,1], linestyle="--", color='black')
l2=ax_roc.plot(fpr_aud_lr,tpr_aud_lr,label='Logistic Regression (PCA):')
l3=ax_roc.plot(svm_aud_fpr2,svm_aud_tpr2,label='SVM(sigmoid) with PCA')
l3=ax_roc.plot(ann_aud_fpr2,ann_aud_tpr2,label='ANN: Config: (128,32) after PCA')
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_title('ROC curves', fontsize=15)
ax_roc.legend(loc=0, prop={'size': 20})
DetCurveDisplay(fpr=fpr_aud_lr, fnr=[1-i for i in tpr_aud_lr], estimator_name='Logistic Regression (PCA):').plot(ax = ax_det)
DetCurveDisplay(fpr=svm_aud_fpr2, fnr=[1-i for i in svm_aud_tpr2], estimator_name='SVM(rbf)').plot(ax = ax_det)
DetCurveDisplay(fpr=ann_aud_fpr2, fnr=[1-i for i in ann_aud_tpr2], estimator_name='ANN: Config: (128,32) after PCA').plot(ax = ax_det)
ax_det.set_title('DET curves', fontsize=15)
ax_det.legend(loc=0, prop={'size': 20})

fig, (ax_roc,ax_det) = plt.subplots(1,2,figsize=(20, 10))
fig.suptitle('Comparing models: ROC and DET for handwritten Data', fontsize=20)
ax_roc.plot([0,1],[0,1], linestyle="--", color='black')
l2=ax_roc.plot(fpr_tel_lr,tpr_tel_lr,label='Logistic Regression (PCA):')
l3=ax_roc.plot(svm_tel_fpr2,svm_tel_tpr2,label='SVM(rbf) with PCA')
l3=ax_roc.plot(ann_tel_fpr2,ann_tel_tpr2,label='ANN: Config:  (128,64,32,32)')
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_title('ROC curves', fontsize=15)
ax_roc.legend(loc=0, prop={'size': 20})
DetCurveDisplay(fpr=fpr_tel_lr, fnr=[1-i for i in tpr_tel_lr], estimator_name='Logistic Regression (PCA):').plot(ax = ax_det)
DetCurveDisplay(fpr=svm_tel_fpr2, fnr=[1-i for i in svm_tel_tpr2], estimator_name='SVM(rbf) with PCA').plot(ax = ax_det)
DetCurveDisplay(fpr=ann_tel_fpr2, fnr=[1-i for i in ann_tel_tpr2], estimator_name='ANN: Config:  (128,32,32,32) with PCA').plot(ax = ax_det)
ax_det.set_title('DET curves', fontsize=15)
ax_det.legend(loc=0, prop={'size': 20})
