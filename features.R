# Feature Engineering of Data Set for Numerai

# Full Training Data
df <- read.table("C:/Users/Anthony Silva/silvat/numerai/numerai_training_data.csv", header=TRUE, sep=",")
df$target <- factor(df$target)

# Modeling Dataset
X <- df[, 4:25]
X[1:21] <- scale(X[1:21])

# Feature Additions from previous analysis
X$feat19sq <- round(X$feature19*X$feature19,5)
X$feat10x19 <- round(X$feature10*X$feature19,5)
X$feat10x8 <- round(X$feature10*X$feature8,5)
X$feat19d10 <- round(X$feature19/X$feature10,5)
X$feat9x19x10x8 <- round(X$feature9 * X$feature19 * X$feature10 * X$feature8,5)
X$feat19x4 <- round(X$feature19*X$feature4,5)
X$feat19x7x4x21 <- round(X$feature19 * X$feature7 * X$feature4 * X$feature21,5)
X$feat4d7 <- X$feature4/X$feature7
X$feat4d7[is.infinite(X$feat4d7)] <- 0


