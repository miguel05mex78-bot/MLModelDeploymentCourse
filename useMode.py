# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 12:11:04 2026

@author: Ana
"""

# PICKLE FILES IN LOCAL ENVIRONMENT
import pickle
#  we will feed the model with the data in numpy array
import numpy as np

#  deserialized in a local file

local_classifier = pickle.load(open('classifier.pickle','rb'))
local_scaler = pickle.load(open('sc.pickle','rb'))

# we predict

new_pred = local_classifier.predict(local_scaler.transform(np.array([[40,20000]])))

new_pred_proba = local_classifier.predict_proba(local_scaler.transform(np.array([[40,20000]])))[:,1]


new_pred_2 = local_classifier.predict(local_scaler.transform(np.array([[42,50000]])))

new_pred_proba_2 = local_classifier.predict_proba(local_scaler.transform(np.array([[42,50000]])))[:,1]
print(new_pred)
print(new_pred_proba)
# this does not know how the model was trainned

