# https://www.udemy.com/course/data-science-and-machine-learning-bootcamp-with-r/learn/lecture/5412904#overview

# To run, source with echo

# -- Import the MASS library, which has a dataset called Boston
if (!require("MASS")) install.packages("MASS")
if (!require("caTools")) install.packages("caTools")
if (!require("neuralnet")) install.packages("neuralnet")
if (!require("ggplot2")) install.packages("ggplot2")

library("MASS")
library("caTools")
library("neuralnet")
library(ggplot2)

data <- Boston

print( "Let's take a peek at the data.")

# What kind of data structure is it?
print( class( data ) )

# What does the data look like?
print( head( data ) )

# What are the column names?
print( names( data ) )

# Make sure there is no blank data
print( any( is.na( data ) ) )

# Goal: We want use the dataset to train a neural net that will predict the medv
# (median value) of homes using the factors in the data set. The medv is a 
# continuous value.

# We want to normalize the data for each factor to be between -1 and +1, this
# helps our minimization function converge faster.
MARGIN = 2 # The value 2 means to apply to the columns
maxs <- apply( data, MARGIN = MARGIN, max )
mins <- apply( data, MARGIN = MARGIN, min )

scaled.data <- scale( data, center=mins, scale=maxs-mins)
scaled <- as.data.frame( scaled.data )

# Split the data into a training and test set
split <- sample.split(scaled$medv, SplitRatio = .7)
train <- subset(scaled,split==T)
test <- subset(scaled,split==F)

# Create a neural net
n <- names( train )
f <- as.formula( paste('medv ~', paste(n[!n %in% "medv"], collapse = ' + ' )))
# f<- as.formula(paste("medv ~" ,paste(n[1:13], collapse = "+")))

nn <- neuralnet(
  f, 
  data = train, 
  hidden = c(5,3),  
  linear.output = T
)

#plot(nn)

predicted.nn.values <- compute(nn, test[1:13])
#str(predicted.nn.values)

true.predictions <- predicted.nn.values$net.result * 
  ( max( data$medv ) - min( data$medv ) ) +
  min( data$medv )
#true.predictions <- predicted.nn.values$net.result*(maxes[14] -mins[14]) + mins[14]

test.r <- test$medv * 
  ( max( data$medv ) - min( data$medv ) ) +
  min( data$medv )

MSE.nn <- sum( ( test.r - true.predictions )^2 ) / nrow( test )

error.df <- data.frame( test.r, true.predictions )

p <- ggplot( error.df,
  aes(
    x=test.r,
    y=true.predictions
  ) ) + geom_point() + stat_smooth()

print( p )
