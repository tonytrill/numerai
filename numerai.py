# -*- coding: utf-8 -*-
"""
Numerai Predicitive Modeling
Tony Silva
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

train = pd.read_csv("C:/Users/Anthony Silva/silvat/numerai/numerai_training_data.csv", index_col="id")
test = pd.read_csv("C:/Users/Anthony Silva/silvat/numerai/numerai_tournament_data.csv", index_col="id")
# Concatenate Datasets to look at data preprocessing
df = pd.concat([train, test], axis=0)

# Data Understanding
# Two data sets given by Numerai
# Training and Test, test set contains validation, test, and live
# validation is used for leaderboard scores, test and live is used for scoring and payouts
# test and live data do not have labels.
# 154,025 rows of total data
# 24 columns, 21 features
# training set contains 108,405 rows
# test set has 45,620 rows
# Of the test set, validation has 16,686 rows, and test 27693, live has 1241
print(df.shape, "total dataframe shape")
print(train.shape, "train dataframe shape")
print(test.shape, "total test dataframe shape")
print(test.loc[test.data_type == "live",:].shape, "live dataframe shape")
print(test.loc[test.data_type == "test",:].shape, "test dataframe shape")
print(test.loc[test.data_type == "validation",:].shape, "validation dataframe shape")

# No Missing Values in the features, only in the target
# Makes sense because we combined training and tests sets in one df
print(df.isnull().sum())

# Create histograms for every column in the dataframe
# Commented out due to slow output
cols = df.columns.values[2:]
#for i in cols:
#    plt.figure()
#    df[i].plot.hist()

sns.pairplot(train[["feature1", "target"]], hue="target")