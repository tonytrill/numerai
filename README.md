# Numerai Machine Learning Project
### by: Tony Silva
### email: tonysilva.ou@gmail.com

## Numerai Machine Learning Challenge
The Numerai Machine Learning Challenge is a relatively new challenge offered by the organization Numerai. The challenge can be found on https://numer.ai. The challenge attempts to bring many different data science minds together to make predictions against the same data set. Much like Kaggle, Numerai has a leader and the opportunity to win money by using Data Science skills.

In the Numerai Challenge, data scientists make predictions against Numerai's encrypted hedge fund data. Numerai then ensemble's the predictions provided by the data scientists in the challenge to generate a meta model. This meta model is then used to make predictions to improve Numerai's hedge fund. From the article, [Super Intelligence for The Stock Market](https://medium.com/numerai/invisible-super-intelligence-for-the-stock-market-3c64b57b244c), "Numerai is able to combine each model into a meta model just like Random Forests combine decision trees into a forest". With this ensemble of predictions Numerai is able to perform better than any one model given to them. They are not looking for "best" overall model but they are looking for many different "good" models that helps with their overall meta model.

Numerai has many different criteria to assess if a data scientist's predictions add to their meta model. 
* Log Loss - On average the model must perform better than -ln(.5) ~ .693.
* Originality - Predictions made must not correlate to other predictions already submitted.
* Concordance - Predictions made on the validation, test, and live data must come from the same model.
* Consistency - The model must perform better than .693 log loss on at least 75% of "eras".

## Machine Learning Approach
In this project both R and Python were utilized. R was utilized more for Feature Engineering, while Python was utilized for the creation of the predictive model. Usually, I would stick with one language for a project, however, for the sake of gaining more experience I utilized both. The machine learning problem for this challenge was binary classification.

### Exploratory Data Analysis
The data given by Numerai came in two different data sets. First, Numerai provided their "Training" data. The training data contained 108,405 observations. The second data set provided contained Numerai's "validation", "test", and "live" data. The validation data was used to utilized to determine position on the leaderboard. The test and live data set were used to assess performance on whether the data scientist received a payout. The training and validation data had target values labeled. The target value was labeled as 0 and 1.

##### Missing Values
Thankfully there were no missing values found in the data set. I utilized the following command to find any.

'''
print(df.isnull().sum())

'''

#### Feature Engineering

### Predictive Modeling
#### Sampling Approach
#### Neural Network Creation
#### Sampling Re-Approach &  K-Nearest Neighbors Classifier

## Final Performance

