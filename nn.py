# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 18:51:32 2017

@author: Anthony Silva
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from keras.models import Sequential
from keras.layers import Dense, Activation
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


#df = pd.read_csv("C:/Users/Anthony Silva/silvat/numerai/numerai_training_data.csv", index_col="id")
df = pd.read_csv("C:/Users/Anthony Silva/silvat/numerai/train.csv", index_col="id")
submit = pd.read_csv("C:/Users/Anthony Silva/silvat/numerai/numerai_tournament_data.csv", index_col="id")
# Concatenate Datasets to look at data preprocessing

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

# Create features for our training and tests sets.
df = create_features(df)

extra_eras = submit.loc[(submit.data_type == "validation") & (submit.era == "era97") 
                       | (submit.era == "era98") | (submit.era == "era99"),]
x_extra_era = extra_eras.iloc[:,3:]
y_extra_era = extra_eras["target"]

train, test = train_test_split(df, test_size=.25, random_state=44)
x_train = train.iloc[:,3:]
y_train = train["target"]

x_train.append(x_extra_era)
y_train.append(y_extra_era)

x_test = test.iloc[:,3:]
y_test = test["target"]

# Standardize the data points
scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# create a submission data set for turning in on Numerai website
final = create_features(submit)
x_submit = final.loc[final.data_type == "validation" ,]
x_submit = x_submit.iloc[:,3:]
y_submit = final.loc[final.data_type == "validation" ,"target"]
final = scaler.transform(final.iloc[:,3:])
x_submit = scaler.transform(x_submit)
model = Sequential()

model.add(Dense(38, activation='sigmoid', input_dim=38))
model.add(Dense(24, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


model.fit(x_train, y_train, epochs=20, batch_size=100, verbose=1)
print()
print(model.evaluate(x_test,y_test, verbose=1))
print(model.evaluate(x_submit, y_submit, verbose=1))
def create_submission(final, submit, model):
    y_pred = model.predict(final)
    submit["probability"] = y_pred
    submit = submit["probability"]
    submit.to_csv("C:/Users/Anthony Silva/silvat/numerai/submission.csv", header=True, sep=",")

create_submission(final, submit, model)
    