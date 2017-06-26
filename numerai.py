# -*- coding: utf-8 -*-
"""
Numerai Predicitive Modeling
Tony Silva
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree
from IPython.display import Image


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
cols = df.columns.values[2:-1]
def plotHists(cols):
   for i in cols:
       plt.figure()
       train[i].plot.hist()
       plt.show()
       plt.close()

# Target has the nearly the same amount for 0 and 1 
print(df.groupby(["target"]).count())
# Boxplots of each feature for the target value. 
# Looking to see if there are any significant differences between target 0 and target 1
# for each of the different features
def plotFeatureTargets():
    for i in cols:
        plt.figure()
        z = "Boxplot of Target Values on " + i
        sns.boxplot(x="target", y=i ,data=train[[i, "target"]]).set_title(z)
        plt.show()
        plt.close()

# Averages for each feature is almost similar.
print(train.groupby(["target"])["feature1"].mean())
print(train.groupby(["target"])["feature1"].std())

# Implement Simple Decision Tree to determine feature interactions
clf = tree.DecisionTreeClassifier()
clf.fit(train[cols], train["target"])

dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=train[cols].feature_names,  
                         class_names=train["target"].target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = pydotplus.graph_from_dot_data(dot_data)  
Image(graph.create_png()) 
