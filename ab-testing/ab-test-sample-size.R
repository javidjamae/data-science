# https://www.statmethods.net/stats/power.html
# https://medium.com/swlh/determining-sample-sizes-for-a-b-testing-using-power-analysis-34719ce9e0e9

library(pwr)

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
samples.per.variant <- round( pwr.2p.test(h=cohens.h, power=power, sig.level=alpha )[[ 'n' ]])

print(samples.per.variant)