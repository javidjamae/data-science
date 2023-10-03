# Linear Binary Classification

These are my notes from [this lecture](https://www.youtube.com/watch?v=DVxR3AwdxoA&list=PLofp2YXfp7TZZ5c7HEChs0_wfEfewLDs7&index=3) that is part of [this course](https://www.cs.utexas.edu/~gdurrett/courses/online-course/materials.html).

Supplementary material:
- [Eisenstein 2.0-2.5, 4.2-4.4.1](https://github.com/jacobeisenstein/gt-nlp-class/blob/master/notes/eisenstein-nlp-notes.pdf)
- [Perceptron and logistic regression](https://www.cs.utexas.edu/~gdurrett/courses/online-course/perc-lr-connections.pdf)

## Lecture Notes

The simplest ML model we'll see in this course is a Linear Binary Classifier. 

What a **classifier** does is take an input a point $\overline{x}$ where

$$ \overline{x} \in \mathbb{R}^d $$

This reads as $x$ is "in" $\mathbb{R}^d$. 

This means that we have numerical data in a $d$-dimensional real-value **feature space**.

We're also going to use the notation:

$$f(\overline{x})$$

Where $f$ is a **feature extractor**.

The reason we think about it this way is because $\overline{x}$ in a lot of our problems is a string (not a numeric value). And $f(\overline{x})$ is a set of **features** in the $d$-dimensional space. 

$$
\overline{x} : \text{string} \rightarrow f(\overline{x}) \in \mathbb{R}^d
$$

There is generally a process of mapping from a string input into a numeric representation that can be used in ML.

Each point (example) has a **label** $y$, and in binary classification there can only be two possibilities:

$$
y \in \{ -1, +1 \}
$$

To visualize this, let's consider a specific case where we have a two-dimensional feature space that can be drawn on a simple x/y graph. For example, think about a situation where the +1's are all in the first quadrant and the -1's are all in the third quadrant. We'll discuss later how we can describe these as being "separable".

> Important Note:
>
> Binary classification isn't always two-dimensional. The term "binary" in binary classification refers to the fact that there are two possible classes or categories that an observation can belong to. The dimensionality, on the other hand, refers to the number of features or variables used to represent each observation. For example, later we'll look at using a bag-of-words feature extractor, which might create a 10000-dimension feature space.

We'll also have a **classifier**, which is a **weight vector** $\overline{w}$. You may also see this written as $\theta$ in some texts. 

The way the weight vector leads to a classification decision happens as follows:

$$
\overline{w}^T \cdot f(\overline{x}) \ge 0
$$

Here we take the dot product of the transpose of our weight vector and the features associated with our point and see if it's greater than 0. 

We can think of the weight vector as being a vector in the same graph described above with the feature points. It points in a direction in the feature space, and the perpendicular to the vector is a **decision boundary**. Any point on the same side of the perpendicular that the vector points will be classified as positive, any point on the other side will be classified as negative. We won't worry about landing directly on the boundary for now. 

We often see:

$$
\overline{w}^T \cdot f(\overline{x}) + b \ge 0
$$

Where $b$ represents a bias.

For the purposes of a Linear Classifier, we'll just introduce the bias as another feature, so for example we can transform the feature vector to have a new value with a bias term of $1$:

$$f(\overline{x}) = [3, -1, 2]$$
$$\downarrow$$
$$\tilde{f}(\overline{x}) = [3, -1, 2, 1]$$

By adding a $1$ to the end, we fold the bias term into the feature vector. This means we don't need to add the $+ b$ and juggle the bias terms in our math, simplifying our equations.

In the next few lectures, we'll talk about the Perceptron and Logistic Regression which are two different algorithms for learning these weights. We'll see that these algorithms are actually quite similar functionally. 