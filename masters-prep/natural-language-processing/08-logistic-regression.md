# Logistic Regression

These are my notes from [this lecture](https://www.youtube.com/watch?v=0naHFT07ja8&list=PLofp2YXfp7TZZ5c7HEChs0_wfEfewLDs7&index=8) that is part of [this course](https://www.cs.utexas.edu/~gdurrett/courses/online-course/materials.html). 

## Lecture Notes

Logistic Regression is a **Discriminitive Probabilistic Model**. This means it places a distribution

$$
P(y|\overline{x})
$$

over labels conditioned on input data points. This is a conditional distribution. 

> Note: The alternative to a Discriminitive Model is a Generative Model that places a joint distribution  $P(\overline{x}, y)$ rather than a conditional distribution. For classification this would include approaches like Naive Bayes. We'll come back to this later in the course.

The probability for the positive label is:

$$
P(y=+1|\overline{x})=\frac{e^{\overline{w}^Tf(\overline{x})}}{1+e^{\overline{w}^Tf(\overline{x})}}
$$

We still have the dot product of the weight and feature vector like we did in the Perceptron algorithm, but now we've embedded it in this **Logistic Function**.

The Logistic Function is

$$
\frac{e^x}{1+e^x}
$$

This gives us a curve where $y$ approaches $1$ as x goes to $+\infty$ and approaches $0$ as x goes to $-\infty$. The intercet on the y-axis is on $0.5$.

This allows us to map from any real number to something that is going to be a valid probability. 

So what this does is takes the $\overline{w}^Tf(\overline{x})$ quantity and turns it into a probability associated with the positive class. 

Because this is a binary logistic regression $P(y=+1|\overline{x})$ and $P(y=-1|\overline{x})$ have to sum to 1. 

So we can deduce that:

$$
P(y=-1|\overline{x})=\frac{1}{1+e^{\overline{w}^Tf(\overline{x})}}
$$

What decision boundary does this imply? When we have a point, we should be able to determine a label. 

$$
\text{return } +1 \text{ if } P(y=+1|\overline{x}) > 0.5
$$

This happens when:
$$
\overline{w}^Tf(\overline{x})>0
$$

You can look at what happens at the origin to confirm that for yourself.

> Note: At the origin ($x = 0$), the value of the logistic function is exactly $0.5$. This corresponds to a balanced situation where the probabilities of both classes are equal. So, when the dot product $\overline{w}^Tf(\overline{x})$ is positive, the logistic function will yield a value greater than $0.5$, suggesting a classification of the positive class. When the dot product is negative, the logistic function will yield a value less than $0.5$, indicating a classification of the negative class.


So the decision rules ends up being the same as the Perceptron, where we're taking a dot product and comparing it to $0$. But with logistic regression we now have a probabilistic interpretation. If you're just doing prediction, this doesn't matter, but where it impacts things is during training. 

### Training

For a labeled dataset:
$$
\bigg(\overline{x}^{[i]}, y^{[i]} \bigg) \vphantom{\sum_{i=1}^{D}} _{i=1}^D
$$

Where we have:
- $D$ is the total number of examples
- $i$ represents the $i$th example
- $\overline{x}^{[i]}$ is each example
- $y^{[i]}$ is each label

We want to maximize the following quantity:

$$
\prod_{i=1}^{D} P(y^{[i]}|\overline{x}^{[i]})
$$

which is the maximum likelihood, or the discriminitive likelihood, of the data. We have a bunch of labels $y$ for data points $\overline{x}$ and we want to maximize the probability of observing this data. 

Because we have a discriminitive model, we're conditioning on $\overline{x}$ and we're maximizing the product of the probability of each $y$ given $\overline{x}$. 

So we set our model parameters $\overline{w}$ such that the data looks as likely as possible. 

We write this as a maximization over $\overline{w}$ and transform the likelihood expression to a log-likelihood:

$$
\max_{\overline{w}}
\sum_{i=1}^{D} \log P(y^{[i]}|\overline{x}^{[i]})
$$

> Note: To simplify the optimization problem, we transform the product of probabilities into a sum of logarithms, which makes the optimization problem more manageable and numerically stable. When dealing with optimization, working with products can be computationally challenging. Additionally, probabilities are often small values, and multiplying many small values together can lead to numerical underflow, which can affect the accuracy of calculations.

The log is a monotonic function. When we want to maximize the likelihood of something (or any function really), we can maximize the log of that function. As long as that function is positive and the log is defined, the maximum remains the same. This simplifies things computationally.

Another transformation is to add a negative sign and flip it around so that instead of maximizing the log likelihood, we're going to minimize the negative log likelihood.

$$
\min_{\overline{w}}
\sum_{i=1}^{D} -\log P(y^{[i]}|\overline{x}^{[i]})
$$

So now we can think of this term:

$$
-\log P(y^{[i]}|\overline{x}^{[i]})
$$

as a loss function in terms of the basic ML principles we've already learned:

$$
\text{loss}(\overline{x}^{[i]}, y^{[i]}, \overline{w})
$$

This gives us our training objective: negative log likelihood (NLL).

Now we can use stochastic gradient descent. So, we need to compute the gradient of each of the terms of the outer sum. So we pick one data point at a time and compute the gradient of the loss:

$$
\frac{\partial}{\partial \overline{w}} \text{loss}(\overline{x}^{[i]}, y^{[i]}, \overline{w})
$$

For the Perceptron we used a simpler algorithmic approach where we adjusted weights. In this case we've formulated a first-principles approach of maximizing the probability of the data and we're going to compute a gradient to directly optimize the objective function (loss function) with gradient descent.

> Note: An objective function is a more general term that encompasses the broader purpose of optimization. It's a function that you want to either maximize or minimize. In machine learning, when you're training a model, you usually want to minimize the loss function. So, the loss function itself can be considered an objective function, but it's a specific type of objective function focused on minimizing prediction errors.


### Logistic Regression Gradients

Assume 
$$
y^{[i]} = +1
$$

We get:

$$
\frac{\partial}{\partial \overline{w}} \text{loss}=\frac{\partial}{\partial \overline{w}} [ -\overline{w}^Tf(\overline{x}) + log(1+ e^{\overline{w}^Tf(\overline{x})})]
$$

When we take the gradient with respect to $\overline{w}$ we get:

$$
=-f(\overline{x})+\frac{1}{1+e^{\overline{w}^Tf(\overline{x})}}\cdot  e^{\overline{w}^Tf(\overline{x})}\cdot f(\overline{x})
$$
$$
=f(\overline{x})\bigg[-1+\frac{e^{\overline{w}^Tf(\overline{x})}}{1+e^{\overline{w}^Tf(\overline{x})}}\bigg]
$$
$$
=f(\overline{x})\bigg[P(y=+1|\overline{x})-1\bigg]
$$


The SGD update is:

$$
\overline{w} \leftarrow \overline{w} - \alpha \frac{\partial}{\partial \overline{w}} \text{loss}(\overline{x}^{[j]}, y^{[j]}, \overline{w})
$$

Which if we add the negative of the gradient, we can write it as:
$$
\overline{w} \leftarrow \overline{w} + \alpha f(\overline{x})\bigg[1 - P(y=+1|\overline{x})\bigg]
$$

If we look at just this part of it:
$$
\overline{w} \leftarrow \overline{w} + \alpha f(\overline{x})
$$

it looks similar to how we handle the update when we  misclassifying a positive as a negative in the Perceptron algorithm, where we add in $f(\overline{x})$ to the weight vector. In logistic regression, we have the same thing, but it's just modulated by the last term in square brackets. 

Let's intuitively think about what happens when:

$$
P(y=+1|\overline{x}) \approx 1
$$

In this case we essentially have the right prediction, so we won't make much of an update because the term where we do $1-P(y=+1|\overline{x})$ will be very close to $0$ so it will cancel out. 

In this case
$$
P(y=+1|\overline{x}) \approx 0
$$

our update will almost look like the Perceptron update because the term where we do $1-P(y=+1|\overline{x})$ will be very close to $1$.

To understand this, we'll let

$$
z = \overline{w}^Tf(\overline{x})
$$

Then we can rewrite the NLL as
$$
\log(1+e^z)-z
$$

That's basically the loss function we had above before we took the gradient. 

If we plot $z$ (x-axis) against the value of the loss (y-axis), we get a shape that kind of looks like the Perceptron loss (linear for negative, 0 for positive). But, with logistic regression it's not as stark of a transition, so it curves and intercepts higher on the y axis, but approaches the same values as Perceptron as x approaches +/- infinity (the limits). 

The two algorithms, which were motivated from very different intuition, end up doing almost exactly the same thing. 

> Note: See below for details on loss equation and gradient when $y=-1$. 


## Further Notes / Research

### Math Explained

#### Deriving Loss Function
Here's a step-by-step explanation of how the loss function is derived:

**Assumption for Binary Classification:**
In binary classification, we have two classes, often labeled as $y = +1$ and $y = -1$. The goal is to predict the probability of an example belonging to the positive class $P(y = +1 | \overline{x})$.

**Model Assumption - Logistic Function:**
Logistic regression assumes that the log-odds of an example belonging to the positive class can be modeled as a linear combination of the features:
$$\log \left( \frac{P(y = +1 | \overline{x})}{1 - P(y = +1 | \overline{x})} \right) = \overline{w}^T \overline{x}$$

> Note: Inside the $\log$ function is the ratio of the probability of a positive outcome (numerator) and the probability of not getting a positive outcome (denominator).

Solving for $P(y = +1 | \overline{x})$:
$$P(y = +1 | \overline{x}) = \frac{e^{\overline{w}^T \overline{x}}}{1 + e^{\overline{w}^T \overline{x}}}$$

**Negative Log-Likelihood Loss:**
The likelihood of observing the training data can be represented as a product of the predicted probabilities for the positive class ($y = +1$) when the actual class is indeed $y = +1$, and the probabilities for the negative class ($y = -1$) when the actual class is $y = -1$. In logarithmic form, this becomes the log-likelihood. To derive the loss function, we negate the log-likelihood and obtain the negative log-likelihood, which we aim to minimize.

**Deriving the Loss Function:**
The negative log-likelihood for a single example can be written as:
$$-\log P(y = +1 | \overline{x}) \quad \text{if } y^{[i]} = +1$$
$$-\log (1 - P(y = +1 | \overline{x})) \quad \text{if } y^{[i]} = -1$$

Substituting the expression for $P(y = +1 | \overline{x})$ derived earlier:
$$-\log \left( \frac{e^{\overline{w}^T \overline{x}}}{1 + e^{\overline{w}^T \overline{x}}} \right) \quad \text{if } y^{[i]} = +1$$
$$-\log \left( \frac{1}{1 + e^{\overline{w}^T \overline{x}}} \right) \quad \text{if } y^{[i]} = -1$$

Simplifying:
$$-\overline{w}^T \overline{x} + \log (1 + e^{\overline{w}^T \overline{x}}) \quad \text{if } y^{[i]} = +1$$
$$\log (1 + e^{\overline{w}^T \overline{x}}) \quad \text{if } y^{[i]} = -1$$

**Combining Positive and Negative Cases:**
The loss function for a single example can be written as:
$$-y^{[i]} \cdot \overline{w}^T \overline{x} + \log (1 + e^{\overline{w}^T \overline{x}})$$

Where $y^{[i]} = +1$ for positive cases and $y^{[i]} = -1$ for negative cases.

**Summing over Examples:**
For a dataset with multiple examples, the total loss is the sum of the losses over all examples. This leads to the loss function you provided:
$$\text{loss} = \sum_{i=1}^D -y^{[i]} \cdot \overline{w}^T \overline{x} + \log (1 + e^{\overline{w}^T \overline{x}})$$

This loss function captures the likelihood-based approach of logistic regression, aiming to maximize the likelihood of observing the training data given the model's predictions. Minimizing this loss function leads to optimizing the model's parameters $\overline{w}$ to make accurate predictions.

#### Partial Derivative and Gradient of the Loss Function for $ y = -1 $

To find the partial derivative for the case when $ y = -1 $, we need to consider the expression for $ P(y=-1|\overline{x}) $ using the original expression for $ P(y=+1|\overline{x}) $.

$$
P(y=-1|\overline{x}) = 1 - P(y=+1|\overline{x}) = \frac{1}{1 + e^{\overline{w}^T f(\overline{x})}}
$$

The loss function when $ y = -1 $ is:

$$
\text{loss} = \log(1 + e^{\overline{w}^T f(\overline{x})})
$$

Taking the partial derivative of this loss function with respect to $ \overline{w} $:

$$
\frac{\partial \text{loss}}{\partial \overline{w}} = \frac{e^{\overline{w}^T f(\overline{x})} \times f(\overline{x})}{1 + e^{\overline{w}^T f(\overline{x})}}
$$

Notice that this expression $ \frac{e^{\overline{w}^T f(\overline{x})}}{1 + e^{\overline{w}^T f(\overline{x})}} $ is the same as $ P(y=+1|\overline{x}) $, so we can rewrite the partial derivative as:

$$
\frac{\partial \text{loss}}{\partial \overline{w}} = P(y=+1|\overline{x}) \times f(\overline{x})
$$

For the negative label case $ y = -1 $, the Stochastic Gradient Descent (SGD) update rule will look similar to the positive case. 

The original SGD update equation is:
$$
\overline{w} \leftarrow \overline{w} - \alpha \frac{\partial}{\partial \overline{w}} \text{loss}(\overline{x}^{[j]}, y^{[j]}, \overline{w})
$$

The gradient for the loss when $ y = -1 $ is:
$$
\frac{\partial \text{loss}}{\partial \overline{w}} = P(y=+1|\overline{x}) \times f(\overline{x})
$$

Plugging this into the original SGD equation, the update rule becomes:
$$
\overline{w} \leftarrow \overline{w} - \alpha \bigg[ P(y=+1|\overline{x}) \times f(\overline{x}) \bigg]
$$

To make it similar to the positive label case, we rewrite it as:
$$
\overline{w} \leftarrow \overline{w} - \alpha f(\overline{x}) \bigg[ 1 - P(y=-1|\overline{x}) \bigg]
$$



#### Log-Odds Explained

Log-odds, also known as the "logit," is a mathematical transformation that takes a probability value and converts it into a different scale. It's often used in statistics and machine learning to work with probabilities in a more manageable way.

In the context of logistic regression, the log-odds is used to model the relationship between the features and the probability of an event happening. Let's break it down:

1. **Probability:** When we talk about probability, we're referring to the chance or likelihood of something occurring. For example, the probability of winning a game or the probability of an email being spam.

2. **Log-Odds (Logit):** The log-odds is the logarithm of the odds of an event happening. Odds, in this context, represent the ratio of the probability of the event occurring to the probability of it not occurring. Taking the logarithm of these odds can be useful because it transforms the probabilities into a scale that is more linear and symmetric.

In simpler terms, the log-odds takes a probability value, which can range from 0 to 1, and transforms it into a value that ranges from negative infinity to positive infinity. This transformed value makes it easier to work with the relationship between features and the event's likelihood.

In the case of logistic regression, the log-odds is used to model how the features influence the probability of a binary outcome (like whether an email is spam or not spam). By combining the features' contributions in a linear way and applying the log-odds transformation, logistic regression tries to find a relationship that can predict the probability of an event happening.


### Logistic Regression Compared to Perceptron

#### Logistic Regression
1. **Model Functionality**: Logistic regression models the probability that the dependent variable belongs to a particular category. It outputs a probability score between 0 and 1.
2. **Activation Function**: It uses the logistic (or sigmoid) function to squash the output between 0 and 1.
3. **Cost Function**: Typically uses log loss or cross-entropy loss, which is differentiable. This leads to a smooth loss landscape that makes optimization easier.
4. **Learning Algorithm**: Gradient-based optimization algorithms like gradient descent are usually employed.
5. **Probabilistic Interpretation**: Outputs can be interpreted as probabilities, providing not just a class label but also information on how confident the model is about that prediction.
6. **Applicability**: Suitable for problems where you need the probabilities or when the decision boundary needs to be more flexible (it can learn nonlinear decision boundaries with added features).

#### Perceptron Algorithm
1. **Model Functionality**: The perceptron algorithm aims to find a hyperplane that separates the classes. It makes hard binary decisions, i.e., outputs are 0 or 1.
2. **Activation Function**: Typically uses a step function to produce binary outputs.
3. **Cost Function**: Uses a loss function that is not differentiable everywhere, leading to a more complex optimization landscape.
4. **Learning Algorithm**: Works by making updates in response to misclassifications, using a simple rule that doesn’t require differentiation.
5. **Probabilistic Interpretation**: No probabilistic interpretation of the outputs, just class labels.
6. **Applicability**: Works best when the data is linearly separable or nearly so. It might fail to converge if the classes are not perfectly separable.

#### Comparison Summary
- **Use Logistic Regression when**:
  - You want probability estimates.
  - Your data might not be linearly separable.
  - You have a large and complex dataset (since logistic regression can be more robust and handle noise better).

- **Use Perceptron when**:
  - You only need hard binary classifications.
  - You have linearly separable data or nearly so.
  - You prefer a simpler, faster algorithm with fewer assumptions.

### Generative Models

A generative model like Naive Bayes learns the joint distribution $P(x, y)$ during the training phase. This means it tries to model how the data was generated in the first place, considering both features and labels. Once trained, it can be used to calculate $P(y | x)$, the conditional probabilities needed for classification.

A generative model is defined by the fact that it models the joint distribution first and then uses it to find the conditional probability $P(y | x)$.

### Conditional vs. Joint Probability

#### Conditional Probability

Conditional probability, denoted as $P(A | B)$, is the probability of event $A$ occurring given that event $B$ has already occurred. In simpler terms, it tells you how likely $A$ is if $B$ is true.

For example, imagine you have a dataset of fruits and their characteristics (like color, shape, etc.). Given a fruit is red ($B$), what's the probability it is an apple ($A$)? This is $P(A | B)$.

In the context of machine learning, when we talk about $P(y | x)$, we are looking at the probability of a particular label $y$ given a set of known features $\bar{x}$.

So, in ML, you use a conditional probability when you already have some information and want to make a prediction or decision based on that. For example, given a patient's symptoms ($\bar{x}$), what's the likelihood they have a certain disease ($y$)?

#### Joint Probability

Joint probability is the probability of multiple events happening together. It is denoted as $P(A \cap B)$ or $P(A, B)$ and describes the likelihood that both $A$ and $B$ occur.

For example, in a dataset of fruits that are either red or green and either apples or bananas, the joint probability $P(\text{Red}, \text{Apple})$ tells us the probability that a fruit picked at random will both be red and an apple.

The term occurs simultaneously means that both events $A$ and $B$ happen at the same time. For example, the joint probability of picking a red card that is also a face card from a deck of cards would be a joint probability.

In the context of ML, you would use a joint probability to understand the overall distribution of data points in your dataset. Generative models like Naive Bayes work by learning the joint probability distribution $P(\bar{x}, y)$ and then using it to calculate $P(y | \bar{x})$.


### Bayes' Theorem

Bayes' theorem is mathematically expressed as:

$$
P(A|B) = \frac{P(B|A) \times P(A)}{P(B)}
$$

- $P(A|B)$: Posterior probability. The probability of $A$ given $B$ is true.
- $P(B|A)$: Likelihood. The probability of $B$ given $A$ is true.
- $P(A)$: Prior probability. The initial probability of $A$ without considering $B$.
- $P(B)$: Marginal likelihood. The total probability of $B$ happening.

**Example**: Let's say you're a doctor diagnosing a rare disease that affects 1% of the population ($P(Disease) = 0.01$, the prior). A test for the disease is 90% accurate ($P(Pos | Disease) = 0.9$). If a random person tests positive ($P(Pos) = ?$), what's the chance they actually have the disease ($P(Disease | Pos)$)?

Using Bayes' theorem:

$$
P(Disease|Pos) = \frac{P(Pos|Disease) \times P(Disease)}{P(Pos)}
$$

Here, $P(Pos)$ can be found through a law of total probability, but for simplicity, let's assume it's 0.08. Then:

$$
P(Disease|Pos) = \frac{0.9 \times 0.01}{0.08} = \frac{0.009}{0.08} \approx 0.1125
$$


