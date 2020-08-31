# Random Selection

library(ggplot2)

# Import the dataset
dataset = read.csv('Ads_CTR_Optimisation.csv')

total_ad_impressions = length(dataset[[1]])
number_of_ads = length(dataset)

# A vector containing the selected ad for each impression
ads_selected = integer(0)

# The cumulative reward for the whole experiment
total_reward = 0

for (n in 1:total_ad_impressions) {
  # Randomly select an ad (in corresponds to column in the dataset)
  ad = sample(1:number_of_ads, 1)
  
  # Append the selected ad to the vector
  ads_selected = append(ads_selected, ad)
  
  # Select from the dataset based on the impression number and ad. Since the 
  #   Reward will be 1 if the given row has a 1 for the column corresponding to the ad
  #   Reward will be 0 if the given row has a 0 for the column corresponding to the ad
  reward = dataset[n, ad]
  
  # Add the reward to the total reward
  total_reward = total_reward + reward
}

# Visualize the results using a bar chart
data = as.data.frame(ads_selected)

# Data & Aesthetics
plot <- ggplot(data, 
  aes(
    x=factor(ads_selected)
  ),
)

# Geometry
plot <- plot + geom_bar(
  color = 'white',
  fill = 'pink',
  alpha = 0.5,
  aes( 
    fill=..count..,
  )
)

# Labels
plot <- plot + labs(
  title = 'Ads Selcted Histogram',
  subtitle = 'Prepared By: Javid Jamae',
  x = 'Ad Selected',
  y = 'Count'
)

plot <- plot + theme(
  plot.title = element_text( hjust = 0.5 ),
  plot.subtitle = element_text( hjust = 0.5 )
)


# Print plot
print( plot )