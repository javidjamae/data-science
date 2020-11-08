B <- F <- c(-1, 1)
design <- expand.grid(B=B, F=F)
B <- design$B
F <- design$F
y <- c(20,21,25,27)
model <- lm(y ~ B*F)
print( summary(model) )

library(pid)
paretoPlot(model)