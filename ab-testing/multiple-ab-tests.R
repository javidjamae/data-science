# http://www.cookbook-r.com/Graphs/Plotting_distributions_(ggplot2)/
# https://www.statmethods.net/stats/power.html
# https://medium.com/swlh/determining-sample-sizes-for-a-b-testing-using-power-analysis-34719ce9e0e9

#set.seed(9384723)

exp.1.control.prob = .1
exp.1.treatment.prob = .15

#########################
# Calculate Sample Size #
#########################
conversion.baseline <- .1
conversion.expected <- .15
power <- .8
alpha <- .05

# Calculate Cohen's h
cohens.h <- ES.h( conversion.expected, conversion.baseline )

# Calculate sample size
samples.per.variant <- round(pwr.2p.test(h=cohens.h, power=power, sig.level=alpha)[['n']])


##################
# Run Experiment #
##################

# experiment.name <- factor(rep(c("Control","Variant"), each=samples.per.variant))

#rating <- c(rnorm(n),rnorm(n))

# Create a binomial distribution where the outcome is 0 or 1
exp.1.control <- c(rbinom(n = samples.per.variant, size=1, prob = exp.1.control.prob))
exp.1.treatment <- c(rbinom(n = samples.per.variant, size=1, prob = exp.1.treatment.prob))

###########
# Plot it #
###########

plot_binomial <- function( dat ) {
  ggplot(dat, aes(x=rating)) +
    geom_histogram(binwidth=.1, colour="black", fill="white") +
    geom_vline(aes(xintercept=mean(rating, na.rm=T)),   # Ignore NA values for mean
               color="red", linetype="dashed", size=1)
}

library(ggplot2)

#ggplot(dat, aes(x=rating)) + geom_histogram(binwidth=.05)

# Plot control as binomial distribution
exp.1.control.frame <- data.frame(
  rating = exp.1.control
)
plot_binomial( exp.1.control.frame )

# Plot treatment as binomial distribution
exp.1.treatment.frame <- data.frame(
  rating = exp.1.treatment
)
plot_binomial( exp.1.treatment.frame )
