# Two rows, positive
#m <- matrix(c(1,2,3,5,15,32), nrow=2, byrow = T)

# Three rows, positive
#m <- matrix(c(1,2,3,5,15,32,4,7,99), nrow=3, byrow = T)

# Four rows, positive and negative
m <- matrix(c(-1,-10,-50,1,2,3,5,15,32,4,7,99), nrow=4, byrow = T)

mins <- apply(m, 2, min)
maxs <- apply(m, 2, max)

# Scale between 0 and 1
#scaled.matrix <- scale(m, center=mins, scale = maxs-mins)

# Scale between -1 and 1
scaled.matrix <- scale(m, center=(maxs-mins) / 2, scale = maxs-mins)

scaled <- as.data.frame(scaled.matrix)

print(scaled)