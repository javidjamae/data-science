# Basics of Learning, Gradient Descent

These are my notes from [this lecture](https://www.youtube.com/watch?v=_We4tlPkaj0&list=PLofp2YXfp7Tbk88uH4jejfXPd2OpWuSLq&index=4). 

We have a set of parameters $\overline{w}$ to optimize. 

These meight be the weights associated with a linear binary classifier, or the parameters of a big neural network.

With supervised learning, we'll assume we have access to labeled data, in this form:

$$(\overline{x}^{[i]}, y^{[i]}) \text{ from } i=i \text{ to } D$$

where:

$\overline{x}^{[i]}$ is the input (e.g. a sentence)

$y^{[i]}$ is a label

$i$ is the ith training example

$D$ is the total number of training examples

Suppose we're doing classificaiton with 10,000 features. 

We can say that $\overline{w}$ is a vector in $\mathbb{R}^{10000}$

$$ \overline{w} \in \mathbb{R}^{10000} $$

In ML, we're searching for an optimal $\overline{w}$. We want to find a weight vector that will allow us to do well in our classification task. 

This is an optimization problem. We need to formulate a **training objective** which is linear over our examples, and optimize it. 

Objective:
$$ \sum_{i=1}^{D} \text{loss}(\overline{x}^{[i]}, y^{[i]}, \overline{w}) $$

By summing up all the losses for each example, this tells us how well our weight vector fits the training data. 

What we want to do is find a $\overline{w}$ that fits this training data very well.

This leads to the idea of stochastic gradient descent. 

There are a lot of algorithms that we can use for optimization.

Stochastic Gradient Descent:
* for $t$ up to num epochs:
  * for $i$ up to $D$:
    * sample $j$ from $[1, D]$
    * $ \overline{w} \leftarrow \overline{w} - \alpha \frac{\partial}{\partial \overline{w}} \text{loss}(\overline{x}^{[j]}, y^{[j]}, \overline{w}) $

The partial derivative is the gradient of the loss with respect to $\overline{w}$. This points towards $\overline{w}$ that give higher loss. So we subtract it to reduce the loss, multiplied by a step size $\alpha$, where $\alpha$ is used to control how far (or fast) we move during gradient descent for each $t$.

For deep neural networks, the step-size parameter $\alpha$ is quite important.

This describes the general framework for Stochastic Gradient Descent. We'll look at binary and logistic regression in subsequent lectures, which will just be instances of this basic framework. 

Even super complicated neural networks will use this same framework. 