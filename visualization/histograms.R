# install.packages('ggplot2')
# install.packages('ggplot2movies')
library(ggplot2)
library(ggplot2movies)

# See column names
#colnames <- colnames(movies)

# See some entries
#head(movies)

# Data & Aesthetics
plot <- ggplot(movies, 
  aes(
    x=rating
  )
)

# Geometry
plot <- plot + geom_histogram(
  binwidth = 0.1,
  color = 'white',
  fill = 'pink',
  alpha = 0.5,
  aes( 
    # fill=..count..,
  )
)

# Labels
plot <- plot + labs(
  title = 'Movie Rating Histogram',
  subtitle = 'Prepared By: Javid Jamae',
  x = 'Movie Rating',
  y = 'Count'
)
plot <- plot + theme(
  plot.title = element_text( hjust = 0.5 ),
  plot.subtitle = element_text( hjust = 0.5 )
)


# Print plot
print( plot )