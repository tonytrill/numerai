# -*- coding: utf-8 -*-
"""
Numerai Predicitive Modeling
Tony Silva
"""

import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats as st

# Using training data provided by Numerai as model creation.
# Using the tournament data as "submit" data.
df = pd.read_csv("C:/Users/Anthony Silva/silvat/numerai/numerai_training_data.csv", index_col="id")
submit = pd.read_csv("C:/Users/Anthony Silva/silvat/numerai/numerai_tournament_data.csv", index_col="id")
x = df.columns.values[2:-1]
y = df.columns.values[-1]


logreg = LogisticRegression(n_jobs=-1, solver="sag")
#scores = cross_val_score(logreg, df[x], df[y], cv=10, scoring="neg_log_loss")
#print(scores.mean())
one_to_left = st.beta(10, 1)  
from_zero_positive = st.expon(0, 50)

params = {  
    "n_estimators": st.randint(3, 40),
    "max_depth": st.randint(3, 40),
    "learning_rate": st.uniform(0.05, 0.4),
    "colsample_bytree": one_to_left,
    "subsample": one_to_left,
    "gamma": st.uniform(0, 10),
    'reg_alpha': from_zero_positive,
    "min_child_weight": from_zero_positive,
}

model = XGBClassifier()

train, test = train_test_split(df, test_size=.3)
gs = RandomizedSearchCV(model, params, n_jobs=-1)  
gs.fit(train[x], train[y])  
print(gs.get_params)
print(gs.best_model_)

