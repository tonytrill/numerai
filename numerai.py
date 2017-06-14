# -*- coding: utf-8 -*-
"""
Numerai Predicitive Modeling
"""

import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import log_loss
training_set = pd.read_csv("C:/Users/Anthony Silva/silvat/numerai/numerai_training_data.csv", index_col="id")
prediction_set = pd.read_csv("C:/Users/Anthony Silva/silvat/numerai/numerai_tournament_data.csv", index_col="id")
val = prediction_set.loc[prediction_set.data_type == "validation",]


training_set = training_set.drop("era", 1)
training_set = training_set.drop("data_type", 1)
val = val.drop("era",1)
val = val.drop("data_type",1)


clf = AdaBoostClassifier()
clf.fit(training_set.iloc[:,:-1], training_set.iloc[:,-1])
predictions = clf.predict(val.iloc[:,:-1])
actual = val.iloc[:,-1]

print(log_loss(actual, predictions))