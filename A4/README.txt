File structure for running the files:
Current folder has the codes.
Put synthetic data file named "19" in the current folder.
Put Image files in the current folder.(coast,forest...)
Put Digit datasets in the current folder.(Names: 1,2,3,4,9)
Put Telugu datasets in the current folder.(Names: bA,chA,dA,lA,tA)

LR.py performs Logistic Regression, with PCA,LDA, stores some ROC values in a file called rocvals.txt. Outputs required accuracies and graphs onto stdout.

KNN_SVM_ANN.py does the remaining tasks. Run it after LR.py only. It also prints accuracies and graphs to stdout. It additionally expects a rocvals.txt file to plot ROCs of LR along with KNN,SVM,ANN.

To run		--> python3 LR.py
Next run     	--> python3 KNN_SVM_ANN.py

