# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 09:32:08 2026

@author: Ana
"""

import numpy as np
import pandas as pd

training_data = pd.read_csv('storepurchasedata.csv')

training_data.describe()
X = training_data.iloc[:, :-1].values

# seleccionar la ultima columna
y = training_data.iloc[:,-1].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =.20,random_state=0)


#scale the data so the results are not influenced by salary that has higher range
# mean is 0 and std is 1
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#classification model using the knn nearest classification techqique
#with  neighbors and minkowski distance
from sklearn.neighbors import KNeighborsClassifier
# minkowski is for ecledian distance
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)


#check the accuracy of the model by predicing
# Model training
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
y_prob = classifier.predict_proba(X_test)[:,1]

# check also the accuracy with the conffusion matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,y_pred))

# our model is 87.5 accurate . The model can predict whether or not a customer
# with a particular age and salary will buy or not with 87.5 accuracy

#  print full description of other metrics: precision, recall, f1 score
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

# predict wether age 40 and salary 20,000 will buy or not
# the prediction is 0 so the customer will no buy
# the model tries to optimize the formula and find the best prediction
new_prediction = classifier.predict(sc.transform(np.array([[40,20000]])))
#in our data the probability is very low that is why the customer will not buy
new_prediction_proba = classifier.predict_proba(sc.transform(np.array([[40,20000]])))[:,1]
print(new_prediction)
print(new_prediction_proba)

# another prediction for ae 45 and salary 50,000
new_pred = classifier.predict(sc.transform(np.array([[42,50000]])))
new_pred_proba = classifier.predict_proba(sc.transform(np.array([[42,50000]])))[:,1]
print(new_pred)
print(new_pred_proba)
# another prediction for ae 45 and salary 50,000 will buy is 1 and he probability is 0.8

# notes:
# to save the model ith pickle; 1) can be in a formula or 2) can be binary so it can be used to predict with new data
# the same scalar has to be used for the model and the prediction


# Picking the Model and Standard Scaler

import pickle

# previously we had the classifier . It will be added in files
model_file = "classifier.pickle"

# wb mode: writing and binary
pickle.dump(classifier, open(model_file,'wb'))

#he scalar
scaler_file = "sc.pickle"

pickle.dump(sc, open(scaler_file,'wb'))








     
     