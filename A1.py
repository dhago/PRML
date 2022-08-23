#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np 
import math

# read the image into a numpy array, change path here
img = mpimg.imread('83.jpg')            

# Perform EVD and plot the frobenius norm between reconstructed and original image
def evd(img):
    A = img.astype('float32')           # Typecast for future calculations
    w,v = np.linalg.eig(A)              # Get eigen values, vectors
    
    # Sort in decreasing order, record indices
    ind = np.argsort(-(np.absolute(w))) 
    
    # Reorder remaining matrices, calculate v inverse
    v = v[:,ind]
    w = np.diag(w[ind])
    vinv = np.linalg.inv(v)
    
    x_val = []
    y_val = []
    k = 0
    while k < 256:
        if k == 255:    #for last value
            if np.iscomplex(w[k,k]):
                break
        # Complex conjugates will be found consecutively, increament k by 2 
        if np.iscomplex(w[k,k]):    
            x_val.append(k+1)
            A_rec = (v[:,:k+1] @ w[0:k+1,0:k+1] @ vinv[:k+1,:]).real
            y_temp = math.sqrt(np.sum((np.square(np.subtract(A,A_rec))))) #fnorm calculation
            y_val.append(y_temp)
            #printImages(img,A_rec,k+2,y_temp,0)
            k+=2
        # Increment k by 1 for real values
        else:
            x_val.append(k)
            A_rec = (v[:,:k] @ w[0:k,0:k] @ vinv[:k,:]).real
            y_temp = math.sqrt(np.sum((np.square(np.subtract(A,A_rec)))))
            y_val.append(y_temp)
            #printImages(img,A_rec,k+1,y_temp,0)
            k+=1
            
    # Plot the Frobenius norm against k
    plt.plot(x_val,y_val, 'r', label = 'EVD')
    plt.xlabel('k-value')
    plt.ylabel('Frobenius Norm')    
    plt.legend()
    
    
# Perform SVD and plot the frobenius norm between reconstructed and original image
def svd(img,interval):
    A = img.astype('float32')               #Typecast for future calcs
    A_t = np.transpose(A)
    AA_t = np.dot(A, A_t)
    '''
        U is calculated as the eigen vector matrix of AA_t
        Sigma as the corresponding singular valued diag matrix
        V_t is calculated as sigma_inverse*U_inverse*A after sorting
    '''
    sigma,U = np.linalg.eig(AA_t)
    sigma = np.sqrt(sigma)
    
    # Sort the matrices in descending order using indices
    ind = np.argsort((np.absolute(sigma)))
    sigma = np.diag(sigma[ind])
    U = U[:,ind]
    
    # V = Sigma_inv @ U_inv @ A
    
    temp = np.dot(np.linalg.inv(sigma), np.linalg.inv(U))
    V_t = np.dot(temp, A)
    
    #Get the x,y values for the plot, inc k by an interval
    x_val = []
    y_val = []
    k = 0
    while k < 256:
        x_val.append(k+1)
        A_rec = (U[:,:k] @ sigma[0:k,0:k] @ V_t[:k,:])
        y_temp = math.sqrt(np.sum((np.square(np.subtract(A,A_rec)))))
        y_val.append(y_temp)
        printImages(img,A_rec,k+1,y_temp,1)
        k+=interval
    plt.plot(x_val,y_val, 'g', label='SVD')
    plt.legend()
    #plt.show()


# Prints Reconstructed and error images
def printImages(og, recon,k, fro_norm, choice):
    fig = plt.figure(figsize=(11,4))
    fig.suptitle('k =' + str(k) + ' Frobenius Norm = ' + str(fro_norm))
    temp = fig.add_subplot(1,2,1)
    imgplt = plt.imshow(recon, cmap='gray')
    temp.set_title('Reconstructed')
    temp = fig.add_subplot(1,2,2)
    imgplt = plt.imshow((og - recon), cmap='gray')
    temp.set_title('Error Image')
    '''Unquote this to save the files instead
    if choice == 0:
        plt.savefig('EVD_imgs/k'+ str(k) + 'e.png', bbox_inches='tight')
    else:
        plt.savefig('SVD_imgs/k'+ str(k) + 's.png', bbox_inches='tight')
    plt.show()'''  

svd(img,1)
evd(img)
