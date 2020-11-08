# Upper Confidence Bound

library(ggplot2)

# Import the dataset
dataset = read.csv('Ads_CTR_Optimisation.csv')

total_ad_impressions = length(dataset[[1]])
number_of_ads = length(dataset)

# A vector containing the selected ad for each impression (1 column, 10,000 rows)
ads_selected = integer(0)

# A vector containing the sum of all the rewards for the given ad (10 columns, 1 row)
sum_of_rewards = integer(number_of_ads)

# A vector containing the count of the times we selected a given ad (10 columns, 1 row)
numbers_of_selections = integer(number_of_ads)

# The cumulative reward for the whole experiment
total_reward = 0

for (n in 1:total_ad_impressions) {
  # Reset the ad and max_upper_bound for each iteration
  ad = 0
  max_upper_bound = 0
  
  for( i in 1:number_of_ads ) {
    # The very first time we run through 
    if( numbers_of_selections[i] > 0 ) {
      average_reward = sum_of_rewards[i] / numbers_of_selections[i]
      delta_i = sqrt(3/2 * log(n) / numbers_of_selections[i])
      upper_bound = average_reward + delta_i
    } else {
      # If the ad has never been 
      upper_bound = 1e400;
    }
    
    # Since ad and max_upper_bound start at 0, this will always get set for the
    #  first ad. Then, if there are any upper bounds that are greater, it will
    #  get updated again.
    # 
    # Since we're setting all the values to 1e400, the first 10 times through 
    #  the outer loop, we will sequentially pick ad 1 through 10 since this
    #  conditional is doing a greater than check.
    if( upper_bound > max_upper_bound ) {
      max_upper_bound = upper_bound
      ad = i
    }
  }
  
  ads_selected = append(ads_selected, ad)
  
  numbers_of_selections[ad] = numbers_of_selections[ad] + 1
  
  # Select from the dataset based on the impression number and ad. Since the 
  #   Reward will be 1 if the given row has a 1 for the column corresponding to the ad
  #   Reward will be 0 if the given row has a 0 for the column corresponding to the ad
  reward = dataset[n, ad]
  
  sum_of_rewards[ad] = sum_of_rewards[ad] + reward
  
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