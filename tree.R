library(rpart)
install.packages("party")
library(party)

df <- read.table("C:/Users/Anthony Silva/silvat/numerai/numerai_training_data.csv", header=TRUE, sep=",")
df$target <- factor(df$target)
df$feature22 <- df$feature21^2
X <- df[, 4:25]

fit <- ctree(target ~ feature1 + feature2 + feature3 + feature4 + feature5 + feature6 + feature7 + feature8
             + feature9 + feature10 + feature11 + feature12 + feature13 + feature14 + feature15 + feature16 + feature17 + feature18
             + feature19 + feature20 + feature21, data=X)

plot(fit)

boxplot(feature22 ~ target, data=df)
