# -*- coding: utf-8 -*-
"""
Build a model that to predict the liklihood of a training sample being 
in the tournament data set.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import log_loss, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
df = pd.read_csv("C:/Users/Anthony Silva/silvat/numerai/numerai_training_data.csv", index_col="id")
submit = pd.read_csv("C:/Users/Anthony Silva/silvat/numerai/numerai_tournament_data.csv", index_col="id")

def create_features(df):
    df["feat19sq"]= df["feature19"]*df["feature19"]
    df["feat10x19"] = df["feature10"]*df["feature19"]
    df["feat10x8"] = df["feature10"]*df["feature8"]
    df["feat9x19x10x8"] = df["feature9"]*df["feature19"]*df["feature10"]*df["feature8"]
    df["feat19x4"] = df["feature19"]*df["feature4"]
    df["feat19x7x4x21"] = df["feature19"]*df["feature7"]*df["feature4"]* df["feature21"]
    df["feat16sq"] = df["feature16"]*df["feature16"]
    df["feat12sq"] = df["feature12"]*df["feature12"]
    df["feat7x13"] = df["feature7"]*df["feature13"]
    df["feat16x12"] = df["feature12"]*df["feature16"]
    df["feat13x4x11"] = df["feature11"]*df["feature13"]*df["feature4"]
    df["feat16x5x6"] = df["feature16"]*df["feature5"]*df["feature6"]
    df["feat15sq"] = df["feature15"]*df["feature15"]
    df["feat11sq"] = df["feature11"]*df["feature11"]
    df["feat5sq"] = df["feature5"]*df["feature5"]
    df["feat15x11x5"] = df["feature15"]*df["feature11"]*df["feature5"]
    df["feat5x10x6"] = df["feature5"]*df["feature10"]*df["feature6"]
    cols = df.columns.values.tolist()
    cols.insert(0, cols.pop(cols.index('target')))
    df = df.reindex(columns=cols)
    return df

def make_response(row):
    if row["data_type"] == "train":
        return 0
    else:
        return 1
# Create features for our training and tests sets.
df = create_features(df)
submit = create_features(submit)
samp = df.sample(50000)
# Over sample Testing Data using sampling with replacement
submit = submit.sample(60000, replace=True)
total = pd.concat([samp,submit], axis=0)
total["response"] = total.apply(lambda row: make_response(row), axis=1)

train, test = train_test_split(total, test_size=.3, random_state=44)
x_train = train.iloc[:,3:-1]
y_train = train["response"]
x_test = test.iloc[:,3:-1]
y_test = test["response"]

model = KNeighborsClassifier(n_neighbors = 100,n_jobs=-1, algorithm="auto")
model.fit(x_train, y_train)

x_test = df.iloc[:,3:]
predictions = model.predict_proba(x_test)
#predict = model.predict(x_test)
#print(log_loss(y_test, predictions))
#print(confusion_matrix(y_test,predict))


preds = []
for j in (i[1] for i in predictions):
    preds.append(j)
preds = np.array(preds)
df["preds"] = preds
df.preds.hist()
df["prediction"] = 0
df.loc[df.preds >.4, "prediction"] = 1
train = df.loc[df.prediction == 1,df.columns.values[:-2]]
train.to_csv("C:/Users/Anthony Silva/silvat/numerai/train.csv")

