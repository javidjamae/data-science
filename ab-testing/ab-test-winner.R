# Looks like Chi-Square and z-Square are the same
# http://rinterested.github.io/statistics/chi_square_same_as_z_test.html

confidence.level = 0.95

control.successes <- 700
control.trials <- 1000

var.a.successes <- 591
var.a.trials <- 1000

total.trials <- control.trials + var.a.trials

# https://www.khanacademy.org/math/statistics-probability/significance-tests-confidence-intervals-two-samples/comparing-two-proportions/v/comparing-population-proportions-1?modal=1
#
# Say that:
# difference.proportion = 0.051
# difference.stddev = 0.217
# confidence.level = 0.95
# alpha = 0.05
# z.score = 1.96
#
# We want the confidence interval that gives us a .95 chance that 
# the true difference in population proportion of control - variation 
# is within some standard error 
# of sample proportion of control - variation (0.051)
# 
# An equivalent statement:
# 
# Confidence that there is a 95% chance that 0.051 is within stderr of 
# population's control - variation
#
# stderr = z.score * difference.stddev
# stderr = 1.96 * 0.217 
#
# H0: difference.proportion = 0 (i.e. control and variation are same, control - variation = 0)
# H1: difference.proportion != 0 (i.e. control and variation are different, control - variation != 0)
# if P( difference.proportion | H0 ) < alpha then reject H0

control.proportion = control.successes / control.trials
control.variance = control.proportion * (1 - control.proportion ) / control.trials

# Proportion (mean) and variance of variation a
var.a.proportion = var.a.successes / var.a.trials
var.a.variance = var.a.proportion * (1 - var.a.proportion) / var.a.trials

difference.proportion = var.a.proportion - control.proportion
difference.variance = control.variance + var.a.variance
difference.stdev = sqrt( difference.variance )

alpha <- 1 - confidence.level
confidence.z.score <- qnorm(1 - alpha / 2, 0, 1)

confidence.stderr <- round( confidence.z.score * difference.stdev, 3 )

print('-----------------------')
print( 
  paste( 
    "There is a", 
    confidence.level, 
    "chance that our sampled conversion rate increase of", 
    difference.proportion, 
    "is within", 
    confidence.stderr, 
    "of the actual conversion rate increase.",
    "The range of possible conversion rate increase is", 
    round( difference.proportion - confidence.stderr, 4 ), 
    "to", 
    round( difference.proportion + confidence.stderr, 4 )
  ) 
)

khan.h0.xbar <- ( control.successes + var.a.successes ) / total.trials
khan.h0.stddev <- sqrt( 
  ( khan.h0.xbar * ( 1 - khan.h0.xbar ) / control.trials ) +
  ( khan.h0.xbar * ( 1 - khan.h0.xbar ) / var.a.trials ) 
)
khan.z.score <- ( difference.proportion - 0 ) / khan.h0.stddev
khan.critical.z <- confidence.z.score

# Couldn't get this to work:
# https://www.cyclismo.org/tutorial/R/pValues.html#calculating-a-single-p-value-from-a-normal-distribution
# khan.pvalue = 2 * ( 1 - pnorm( khan.h0.xbar, mean=0, sd=khan.h0.stddev ) )

print('-----------------------')
print(
  paste(
    "Using Khan Academy method, this test IS",
    if ( khan.z.score > khan.critical.z ) "statistically significant." else "NOT statistically significant.",
    "There is a",
    alpha / 2,
    "chance that we sample a z-statistic greater than",
    round( khan.critical.z, 3 ),
    "and we got",
    round( khan.z.score, 3 )
  )
)

# Chi-Square derived p-value (using Pearson's Chi Square test)
# https://www.evanmiller.org/ab-testing/chi-squared.html

print('-----------------------')
M <- as.table(
  rbind(
    c(control.successes, control.trials - control.successes), 
    c(var.a.successes, var.a.trials - var.a.successes)
  )
)
dimnames(M) <- list( 
  variation = c("control", "variation a"), 
  conversion = c("success", "failure")
)
#print(M)

chi.square <- chisq.test(M, correct=F)
chi.p.value = chi.square$p.value

print(
  paste(
    "Using Chi-Square method, this test is",
    if ( chi.p.value < alpha ) "statistically significant." else "not statistically significant.",
    "chi.p.value:",
    round( chi.p.value, 3 ),
    "alpha:",
    round( alpha, 3 )
  )
)

print('-----------------------')

# http://www.sthda.com/english/wiki/two-proportions-z-test-in-r
print ( 
  prop.test(
    x=c( control.successes, var.a.successes ),
    n=c( control.trials, var.a.trials ), 
    conf.level=confidence.level,
    p = NULL, 
    alternative = "two.sided",
    correct = F
  )
)

print('-----------------------')

########################
# Confidence Intervals #
########################
# https://rcompanion.org/handbook/H_02.html
library(DescTools)

control.confidence = BinomCI(
  control.successes, 
  control.trials,
  conf.level = confidence.level,
  method = "wilson"
)

print( "Control confidence interval:" )
print( control.confidence )

var.a.confidence = BinomCI(
  var.a.successes, 
  var.a.trials,
  conf.level = confidence.level,
  method = "wilson"
)

print( "Variation A confidence interval:" )
print( var.a.confidence )
