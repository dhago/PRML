*******************PART 1*********************
The P1.py file contains the code that utilises K-means, GMMs to classify the data
For part 1, make sure that all the Image data points and Synthetic data points are in the same folder as that of the .py file.

I have loaded the files using a function data loader that requires you to fill in the correct FOLDER PATH/FILE NAME (whichever is applicable) to the data, They are in lines 1087 and 1145 onwards.

for synthetic data make sure to do the same. They are in lines 51 and 75. 

Basically where ever there is an "open(filename)" or a dataloader kind of operation.

Running the code takes a while so make sure to comment out anything that take too long to run. (the entire code will definitely finish but takes time due to iterative gmm calculations)

*******************PART 2*********************
The given "P2" folder contains all the codes required, but the data set folders have to be added IN THE SAME FOLDER as that of the codes
To create such folder:
	Keep all 4 code files in a folder. Keep the HMM-Code folder ,data set folders in this same folder (files that had to be added in our case was "1,2,3,4,9,bA,chA,dA,IA,tA,HMM-CODE" + the 4 .py files that are in the "P2" folder)

dtwTel.py runs DTW on Telugu Characters assigned to our team
hmmTel.py runs HMM on Telugu Characters assigned to our team
To compare ROC_DET curves for these two, first run dtwTel.py, then uncomment last part in hmmTel.py and run it

dtwAud.py runs DTW on Audio data sets assigned to our team
hmmAud.py runs HMM on Audio data sets assigned to our team
To compare ROC_DET curves for these two, first run dtwAud.py, then uncomment last part in hmmAud.py and run it

Note that DTW takes long time to run, it produces the confusion matrix and ROC,DET curves and prints Accuracy.
HMM runs quicker, still takes a while because of the custom KMeans function used instead of the inbuilt Kmeans functions.
HMM codes give out, Confusion matrices and ROC,DET curves for each of the cases. IGnore Prints
Last PRINT gives accuracies
