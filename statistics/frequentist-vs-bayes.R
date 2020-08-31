# In this example, I'm going to simulate flipping different coins:
# - .5 Heads / .5 Tails
# - .51 Heads / .49 Tails
# - .6 Heads / .4 Tails
# - .8 Heads / .2 Tails
# 
# I'll use three different method:
# - Frequentist - Fixed Horizon test (no corrections)
# - Frequentist - Sequential Sampling (Evan Miller formula)
# - Bayesian - No priori
# - Bayesian - Priori of 50%, assuming fair coin
#
# I want to learn:
# - Which method gives me a significant result the fastest
# - Which has the lowest error rate
# - 
# References
# - https://tinyheero.github.io/2017/03/08/how-to-bayesian-infer-101.html
# - https://www.analyticsvidhya.com/blog/2016/06/bayesian-statistics-beginners-simple-english/
# - https://nbviewer.jupyter.org/github/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/blob/master/Chapter1_Introduction/Ch1_Introduction_PyMC3.ipynb
# - https://xcelab.net/rm/statistical-rethinking/
# - https://www.dynamicyield.com/lesson/bayesian-testing/
# - https://www.chrisstucchio.com/blog/2015/dont_use_bandits.html