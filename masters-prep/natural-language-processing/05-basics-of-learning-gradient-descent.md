# Basics of Learning, Gradient Descent

These are my notes from [this lecture](https://www.youtube.com/watch?v=_We4tlPkaj0&list=PLofp2YXfp7TZZ5c7HEChs0_wfEfewLDs7&index=5) that is part of [this course](https://www.cs.utexas.edu/~gdurrett/courses/online-course/materials.html). 

## Lecture Notes

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

The partial derivative is the gradient of the loss with respect to $\overline{w}$. This points towards $\overline{w}$ that give higher loss. So we subtract it to reduce the loss, multiplied by a step size (or learning rate) $\alpha$, where $\alpha$ is used to control how far (or fast) we move during gradient descent for each $t$.

For deep neural networks, the step-size parameter $\alpha$ is quite important.

This describes the general framework for Stochastic Gradient Descent. We'll look at binary and logistic regression in subsequent lectures, which will just be instances of this basic framework. 

Even super complicated neural networks will use this same framework. 

## Dimensions explained

1. $f(x)$: In the context of the perceptron, $f(x)$ is typically a function that takes a single input example $x$ and maps it to a feature vector. The result of $f(x)$ for a given example is a vector of feature values.

2. $f(x_i)$: This would refer to the feature vector for the $i$-th example in your dataset. It's still a vector, containing the feature values for that specific example.

3. $D$: When you're dealing with an entire dataset of $D$ examples, you might represent it as a matrix, where each row is the feature vector for a particular example. So the matrix would have dimensions $D \times n$, where $D$ is the number of examples, and $n$ is the number of features.

4. $w$: The weight vector $w$ in the context of the perceptron algorithm is a vector containing the weights associated with each feature. It has the same number of elements as there are features, so its dimension would be $n$, the same as the length of a feature vector.

**Summary**

- $f(x)$: Function mapping an input example to a feature vector.
- $f(x_i)$: Feature vector for the $i$-th example.
- Dataset: Matrix representing all examples, with dimensions $D \times n$.
- $w$: Weight vector with dimensions $n$.

The choice of representing these as vectors or matrices can sometimes depend on the specific implementation or the particular variant of the algorithm, but the above descriptions are typical for the standard perceptron algorithm.

