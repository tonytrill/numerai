# Numerai Machine Learning Project
### by: Tony Silva
### email: tonysilva.ou@gmail.com

## Numerai Machine Learning Challenge
The Numerai Machine Learning Challenge is a relatively new challenge offered by the organization Numerai. The challenge can be found on https://numer.ai. The challenge attempts to bring many different data science minds together to make predictions against the same data set. Much like Kaggle, Numerai has a leaderboard and the opportunity to win money by using Data Science skills.

In the Numerai Challenge, data scientists make predictions against Numerai's encrypted hedge fund data. Numerai then ensemble's the predictions provided by the data scientists in the challenge to generate a meta model. This meta model is then used to make predictions to improve Numerai's hedge fund. From the article, [Super Intelligence for The Stock Market](https://medium.com/numerai/invisible-super-intelligence-for-the-stock-market-3c64b57b244c), "Numerai is able to combine each model into a meta model just like Random Forests combine decision trees into a forest". With this ensemble of predictions Numerai is able to perform better than any one model given to them. They are not looking for "best" overall model but they are looking for many different "good" models that helps with their overall meta model.

Numerai has many different criteria to assess if a data scientist's predictions add to their meta model. 
* Log Loss - On average the model must perform better than -ln(.5) ~ .693.
* Originality - Predictions made must not correlate to other predictions already submitted.
* Concordance - Predictions made on the validation, test, and live data must come from the same model.
* Consistency - The model must perform better than .693 log loss on at least 75% of "eras".

## Machine Learning Approach
In this project both R and Python were utilized. R was utilized more for Feature Engineering, while Python was utilized for the creation of the predictive model. Usually, I would stick with one language for a project, however, for the sake of gaining more experience I utilized both. The machine learning problem for this challenge was binary classification, predicting 0 or 1 on a target variable.

### Exploratory Data Analysis
The data given by Numerai came in two different data sets. First, Numerai provided their "Training" data. The training data contained 108,405 observations. The second data set provided contained Numerai's "validation", "test", and "live" data. This data set was called their "Tournament Data". The validation data was used to determine position on the leaderboard. The test and live data set were used to assess performance on whether the data scientist received a payout. The training and validation data had target values labeled. The target value was labeled as 0 and 1. In the data set both target values had roughly the same number of observations.

```
print(df.groupby(["target"]).count())
```

| Target        | Count         |
| :-----------: |:-------------:|
| 0             | 62122         |
| 1             | 62969         |

Each of the data sets provided an id column, which labeled each observation; a data_type column, specifying what type of observation it was, either train, validation, test or live; an era column, where the era specified was the time frame the observation was taken from. The challengers were told this column should not be utilized as a feature and the time frame between eras was not specified nor the distinction of what an era actually is. The data sets provided 21 features, labeled "feature1", "feature2" ... "feature21".

#### Missing Values
Thankfully there were no missing values found in the data set. I utilized the following command in attempts to find any missing data.
```
print(df.isnull().sum())
```

All of the features' values fall between the range of [0,1]. For all of the features, the distributions appear to follow a normal distribution. The density plot of feature1 describes pretty well what is seen across all features, however, with different variability and size of the "bell" in the bell curve. An example can be seen below:

![feature1 distribution](/images/distribution.jpg)

Density distributions for each target value were observed for each of the features. This could help determine if the different target values followed a different distribution for any of the features.

```
names <- colnames(X)
for (i in 1:(dim(X)[2]))
{
  if (names[i] != "target")
  {
    plot(density(X[X$target == 1,i]), main = names[i])
    lines(density(X[X$target == 0, i]))
  }
}
```

![feature6 density](/images/density.jpg)

I would have hoped to have seen two different distinct normal curves. This would have told me that for a given feature we could derive a differing distribution between the target values. This could have helped generate our predictive model. However, for each feature, the target values followed the same distribution. The next approach was to determine correlations between features.

```
library(ggplot2)
library(reshape2)
qplot(x=Var1, y=Var2, data=melt(cor(X[, !(names(X) %in% "target")])), fill = value, geom="tile")
```

or

```
plt.figure()
sns.heatmap(train.corr())
plt.show()
plt.close()
```

Produces the following Correlation Matrix.

![correlation matrix](/images/correlations.png)

#### Feature Engineering
Unfortunately the data set was entirely encrypted and features were unnamed. This meant applying intuition around the project was impossible and generating new features would be a challenge. In order to determine important features and important interactions between features, I used a combination of a classic decision tree and XGBoost Decision Trees.

For the simple decision tree:

```
library(rpart)
library(rpart.plot)
libaray(rattle)
# Generate simple decision tree to determine if any variables have interactions 
fit <- rpart(target ~ . , data=X , method="class", control=rpart.control(minsplit=1000, minbucket=1, cp=0.001))
fancyRpartPlot(fit, sub = "")
```
The visualization that was generated was then used to locate potential important features and important interactions.

![simple decision tree](/images/simple_decision_tree.png)

Important features were identified as those that repeat multiple times in the tree as a whole or on specific branches of the tree. Important interactions were identified as those features that reoccurred on branches together. I was then able to come up with additional features by feature transformation by squaring or cubing features that were found to be important and multiplying important interactions together.

A more extensive approach was taken using XGBoost. I first looked at only the 1-depth trees, then 2-depth trees, all the way to 5 depth trees. At each step in the analysis I looked for important features and important feature interactions. I made sure to check myself at the deeper depth trees to determine if any feature occurrences seemed to match the previous depth trees. That way I could justify creating new features.

and for XGBoost:

```
library(xgboost)
fit <- xgb.train(data=trmat, label=y, max.depth=1, eta=1,nthread=2,nrounds = 5, eval.metric = "logloss", objective="binary:logistic")
xgb.plot.tree(model = fit)
# Ran depth 2,3,4,5 trees to find reoccuring important interactions
fit <- xgb.train(data=trmat,label=y, max.depth=5, eta=1,nthread=2,nrounds = 2, eval.metric = "logloss", objective="binary:logistic")
xgb.plot.tree(model = fit)
pred <- predict(fit, newdata=temat)
importance_matrix <- xgb.importance(model = fit)
xgb.plot.importance(importance_matrix=importance_matrix)
```

Additionally the XGBoost package in R provides an easy way to measure true feature importance for the tree. By doing so, I was able to not only generate new features visually but also quantitatively. 

Through this methodology I was able to generate new features in attempt to help aid the generation of a good predicitive model.

### Predictive Modeling

In the predictive modeling step of the project I began with generating a basic logistic regression model. However, by the Numerai standards this model was not considered "good" because it failed the originality step of Numerai's model assessment. Since there are many predictions using logistic regression, there predictions are highly correlated and fail the orginality step. I decided I would go with building a Neural Network in attempts to pass this step in Numerai's measurements.

#### Sampling Approach

At first I split Numerai's training data into my own training and testing sets. I used 75% of the data as training with the rest as testing data. Numerai's dataset was then used to measure leaderboard performance. So in reality the Numerai "validation" set was my test set, while the testing set I made was actually a validation set used to not overfit the data.

#### Neural Network Creation

In order to create an efficient Nueral Network, I utilized the Keras Python library. The Keras library makes it very easy and efficient to build a Neural Network and get it up and running quickly. Along with Keras, I utilized Kera's compability with TensorFlow as a backend. After, scaling the data and splitting into my training and testing sets I created my Neural Network as follows:

```
from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential()

model.add(Dense(38, activation='sigmoid', input_dim=38))
model.add(Dense(24, activation='sigmoid'))
model.add(Dense(12, activation='sigmoid'))
model.add(Dense(8, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=20, batch_size=100, verbose=1)
```

The Neural Network was created with Keras' Sequential Model. The NN contains an input layer, output layer and three hidden layers of various sizes. I wanted to keep the NN as basic as possible so as not to over fit the training data but still perform well enough in Numerai's eyes. The NN utilized log loss as the measure to minimize and fitted the model in batches of 100 samples over 20 iterations through the data. The fitted model performed well but not good enough.

![nn performance](/images/performance1.PNG)

As you can see, the model performed better than random guessing overall, however, it only performed better than random guessing on 66.66% of the eras in the leaderboard data. So Numerai considers my data not "consistent". I attempted to tune my model, however, even though I would reduce my overall log loss my consistency did not improve. So I was performing really well on some eras but not on others.

#### Sampling Re-Approach & K-Nearest Neighbors Classifier

In order to mitigate consistency issue I took two different approaches. First, I built a K-Nearest Neighbors Classifier to find training samples that are most like the samples in Numerai's leaderboard data. This included creating a different target variable as 0 in case a sample is in the training and 1 if the data resides in the Numerai leaderboard data. I then ran the classifier against the full training data and only selected those samples at a threshold.

Split at .4

![performance2](/images/performance2.PNG)

If splitting out too much of the data the model ends up performing worse than before and performs poorly on consistency.

Split at .5

![bad performance](/images/badperformance.5.PNG)

Second, I looked at which eras were performing the worst in the leaderboard data. Of those eras I added them to the training data. My hopes would be the model would not be swayed to much by these samples but just enough to promote the consistency measure in the model. I felt this was the right method to use because there were only a few thousand samples compared to the 75,000 true training samples .

## Final Performance

After performing the methods in the previous section I was able to get the model to perform more consistent and meet all of Numerai's criteria.

![performance3](/images/performance3addingvalidations.PNG)

Now that my model is in contention to add to Numerai's meta model, I now have to wait for Numerai to score the "live" data in order to be considered for a payout.

Overall, this was a great experience working with a brand new data set and applying Machine Learning Methods.
