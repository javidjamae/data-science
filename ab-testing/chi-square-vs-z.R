confidence.level = 0.95

a.successes <- 188
a.trials <- 2069

b.successes <- 591
b.trials <- 1000

res.prop.test <- prop.test(
  x=c( a.successes, b.successes ),
  n=c( a.trials, b.trials ), 
  conf.level=confidence.level,
  p = NULL, 
  alternative = "two.sided",
  correct = F
)

print( res.prop.test )

res.fisher.test <- fisher.test(alternative = 't',
  rbind(c(3,9),c(13,4))
)

print( res.fisher.test )