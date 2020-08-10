# install.packages('ggplot2')
# install.packages('ggplot2movies')
library(ggplot2)

df <- mtcars

## Data & Aesthetics
plot <- ggplot(
  df,
  aes(
    x = wt,
    y = mpg
    # y = factor( cyl ) 
  )
)

## Labels
plot <- plot + labs(
  title = 'MT Car Scatterplot',
  subtitle = 'Prepared By: Javid Jamae'
#  x = 'Weight',
#  y = 'Miles Per Gallon'
)
plot <- plot + theme(
  plot.title = element_text( hjust = 0.5 ),
  plot.subtitle = element_text( hjust = 0.5 )
)


## Geometry

# Set size in the data level
plot.basic <- plot + geom_point(
  size=5,
  alpha=0.5,
)

#print( plot.basic )

# Set size based on a third factor, using aesthetics (aes)
plot.by.hp <- plot + geom_point(
  aes(
    size = hp,  # it knows that this is the field name from my data frame (df)
    # size = factor( cyl ) # cylinders are even numbers, so just just show existing factors
    # size = disp
    # size = drat
    # size = hp
    # size = qsec
    # size = wt
    shape=factor( cyl ),
    color=factor( cyl )
  )
)

print( plot.by.hp )