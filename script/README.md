---------------------------------------------------------------
In this README, model for Machine Learning
algorithms in Handshake Analytics has been shown
---------------------------------------------------------------

How to run:
You can run using the command line with python ml.py
To view weight plots from logistic regression, uncomment
line 198 which calls ml and comment out line 201 which is
also calling ml

---------------------------------------------------------------
Model explanation:

run_models -> is the driver method where subset of features
are selected.

ml -> prepares the data for machine learning and  calls the 
models. By default it runs KNN, Logistic, and Decision Tree 
models

line 36-43: Uncomment these lines to get a balanced dataset

For each of the model, random set of hyper parameters are given
with the variable c_l
The data is also normalized in this section
K-Fold cross validation is applied, and a development(dev) set is 
selected for every training set. Dev set is selected with 1/4th of
the training set. This is used to select the best hyper-parameter for
training the model. In the final training for the model, the entire
training set is used.
The trained model is then tested on the untouched test set. After 
all the folds (10) are done, the average training and test accuracies
are taken as the final prediction rates. F1 scores has been used as well
as a evaluation metric.
--------------------------------------------------------------------
