# Perceptron

These are my notes from [this lecture](https://www.youtube.com/watch?v=tMGv5ZcuVP4&list=PLofp2YXfp7Tbk88uH4jejfXPd2OpWuSLq&index=5). 

## Lecture Notes

We'll introduce the **perceptron**, a basic algorithm for training a linear binary classifier. We'll apply this classifier to our sentiment analysis example. 

A binary classifier has the following decision rule:

$$ \overline{w}^T \cdot f(\overline{x}) > 0 $$

Where:

* $ \overline{w} $ is the weight vector
* $ \overline{w}^T $ is the transposed weight vector
* $ f(\overline{x}) $ is the feature vector 


If the dot product of the weights and the features is greater than 0, then $y = +1$; otherwise it is $y = -1$. Or:

$$ y \in \{-1, +1\} $$ 

### Perceptron Algorithm
Here is how we would implement the perceptron algorithm for linear binary classification:

* initialize the weight vector $\overline{w}$
* for $t$ up to num epochs:
  * for $i$ up to $D$:
    * $y_{\text{pred}} = \text{sign}(\overline{w}^T \cdot f(\overline{x}^{[i]}))$
    * $\overline{w} \leftarrow \overline{w} + \alpha y^{[i]} f(\overline{x}^{[i]})$ if $y_{\text{pred}} \neq y^{[i]}$

* $\overline{w} \leftarrow \overline{w} + \alpha y^{[i]} f(\overline{x}^{[i]})$ if $y_{\text{pred}} \neq y^{[i]}$

Where:
* $y_{\text{pred}}$ is our predicted $y$ classification value
* $y_{\text{pred}} = y^{[i]}$ means we predicted the example right and we don't update the weights
* $\alpha$ is the step size (or learning rate) and is a hyperparameter used by the algorithm while training the model.
* $D$ is the total number of training examples
* the sign function will return $+1$ if its argument is greater than $0$, and 
$−1$ if its argument is less than or equal to $0$.
* The weight update function is the partial derivative of the loss function (math explained below)

> Note:
>
> In the actual lecture video, he writes out the weight update logic in a much more verbose way:
>
> * $ \overline{w} \leftarrow \overline{w}$ if $y_{\text{pred}} = y^{[i]}$
>   * else $\overline{w} \leftarrow \overline{w} + \alpha f(\overline{x}^{[i]})$ if $y^{[i]} = +1$
>   * else $\overline{w} \leftarrow \overline{w} - \alpha f(\overline{x}^{[i]})$ if $y^{[i]} = -1$

Remember that the features and the weights are in the same space. If we have a 10000 dimension feature space, we also have a 10000 dimension weight vector, so they can be added together. 

The positive case encourages the dot product of the weights and the features to be more positive on future iterations. The negative case encourages the dot product of the weights and the features to be more negative on future iterations. This will converge if the data are eventually separable. 

### Example 1: Converging
Let's look at an example where we have the following examples:

| movie review | y | $f(\overline{x})$: mgbn |
|--------------|---|-------------------------|
| movie good   | +1| [1, 1, 0, 0]            |
| movie bad    | -1| [1, 0, 1, 0]            |
| not good     | -1| [0, 1, 0, 1]            |

`mgbn` represents a bag-of-words feature space vector. We encode each example during the feature extraction phase based on the count of the words in our reviews: **m**ovie, **g**ood, **b**ad, **n**ot.

So, `movie good` gets encoded as `1100` because it has `movie` and `good` in the review. Or, in vector form, it could be represented as $[1, 1, 0, 0]$

We'll start by setting our weight vector to $\overline{w} = [0, 0, 0, 0]$ and setting $\alpha = 1$ (common learning rate).

We can look at the first example and calculate:

$ \overline{w}^T \cdot f(\overline{x}) = 0 \cdot 1 + 0 \cdot 1 + 0 \cdot 0 + 0 \cdot 0 = 0 $


Since we are calculating the predicted value as $y_{\text{pred}} \leftarrow \overline{w}^T \cdot f(\overline{x}^{[i]}) > 0$, this results in:

$y_{\text{pred}} = 0 > 0 \Rightarrow y_{\text{pred}} = -1$

Given that the actual label for the first example is $y^{[1]} = +1$, this prediction is incorrect. According to the given update rules, we need to update the weights as:

$\overline{w} \leftarrow \overline{w} + \alpha f(\overline{x}^{[1]}) \text{ since } y^{[1]} = +1$

Given that $\alpha = 1$, the weights are updated to:

$\overline{w} \leftarrow [0, 0, 0, 0] + 1 \cdot [1, 1, 0, 0] = [1, 1, 0, 0]$

Now, we move on to the second example:

Feature vector: $f(\overline{x}^{[2]}) = [1, 0, 1, 0]$ and $y^{[2]} = -1$.

Calculate $y_{\text{pred}}$:

$\overline{w}^T \cdot f(\overline{x}^{[2]}) = 1 \cdot 1 + 1 \cdot 0 + 0 \cdot 1 + 0 \cdot 0 = 1$

$y_{\text{pred}} = 1 > 0 \Rightarrow y_{\text{pred}} = +1$

Again, the prediction is incorrect, so we update the weights:

$\overline{w} \leftarrow \overline{w} - \alpha f(\overline{x}^{[2]}) \text{ since } y^{[2]} = -1$

Updated weights:

$\overline{w} \leftarrow [1, 1, 0, 0] - 1 \cdot [1, 0, 1, 0] = [0, 1, -1, 0]$

Here is a full run through the algorithm until it converges. We'll start with an initial weight vector of: $\overline{w} = [0, 0, 0, 0]$


**Epoch 1:**
1. First example: $f(\overline{x}^{[1]}) = [1, 1, 0, 0]$, $y^{[1]} = +1$
    - $\overline{w} = [0, 0, 0, 0]$
    - $ \overline{w}^T \cdot f(\overline{x}^{[1]}) = 0 \cdot 1 + 0 \cdot 1 + 0 \cdot 0 + 0 \cdot 0 = 0$
    - $y_{\text{pred}} =  \overline{w}^T \cdot f(\overline{x}^{[1]}) > 0 = -1$
    - $\overline{w}_{\text{new}} = \overline{w} + \alpha \cdot f(\overline{x}^{[1]}) = [0, 0, 0, 0] + 1 \cdot [1, 1, 0, 0] = [1, 1, 0, 0]$

2. Second example: $f(\overline{x}^{[2]}) = [1, 0, 1, 0]$, $y^{[2]} = -1$
    - $\overline{w} = [1, 1, 0, 0]$
    - $\overline{w}^T \cdot f(\overline{x}^{[2]}) = 1 \cdot 1 + 1 \cdot 0 + 0 \cdot 1 + 0 \cdot 0 = 1$
    - $y_{\text{pred}} =  \overline{w}^T \cdot f(\overline{x}^{[2]}) > 0 = +1$
    - $\overline{w}_{\text{new}} = \overline{w} - \alpha \cdot f(\overline{x}^{[2]}) = [1, 1, 0, 0] - 1 \cdot [1, 0, 1, 0] = [0, 1, -1, 0]$

3. Third example: $f(\overline{x}^{[3]}) = [0, 1, 0, 1]$, $y^{[3]} = -1$
    - $\overline{w} = [0, 1, -1, 0]$
    - $\overline{w}^T \cdot f(\overline{x}^{[3]}) = 0 \cdot 0 + 1 \cdot 1 - 1 \cdot 0 + 0 \cdot 1 = 1$
    - $y_{\text{pred}} =  \overline{w}^T \cdot f(\overline{x}^{[3]}) > 0 = +1$
    - $\overline{w}_{\text{new}} = \overline{w} - \alpha \cdot f(\overline{x}^{[3]}) = [0, 1, -1, 0] - 1 \cdot [0, 1, 0, 1] = [0, 0, -1, -1]$


**Epoch 2:**

Start next epoch:

4. First example: $f(\overline{x}^{[1]}) = [1, 1, 0, 0]$, $y^{[1]} = +1$
    - $\overline{w} = [0, 0, -1, -1]$
    - $ \overline{w}^T \cdot f(\overline{x}^{[1]}) = 0 \cdot 1 + 0 \cdot 1 - 1 \cdot 0 - 1 \cdot 0 = 0$
    - $y_{\text{pred}} =  \overline{w}^T \cdot f(\overline{x}^{[1]}) > 0 = -1$
    - $\overline{w}_{\text{new}} = \overline{w} + \alpha \cdot f(\overline{x}^{[1]}) = [0, 0, -1, -1] + 1 \cdot [1, 1, 0, 0] = [1, 1, -1, -1]$


5. Second example: $f(\overline{x}^{[2]}) = [1, 0, 1, 0]$, $y^{[2]} = -1$
    - $\overline{w} = [1, 1, -1, -1]$
    - $\overline{w}^T \cdot f(\overline{x}^{[2]}) = 1 \cdot 1 + 1 \cdot 0 - 1 \cdot 1 - 1 \cdot 0 = 0$
    - $y_{\text{pred}} =  \overline{w}^T \cdot f(\overline{x}^{[2]}) > 0 = -1$
    - $y_{\text{pred}} = y^{[2]}$, so $\overline{w}_{\text{new}} = \overline{w} = [1, 1, -1, -1]$

6. Third example: $f(\overline{x}^{[3]}) = [0, 1, 0, 1]$, $y^{[3]} = -1$
    - $\overline{w} = [1, 1, -1, -1]$
    - $\overline{w}^T \cdot f(\overline{x}^{[3]}) = 1 \cdot 0 + 1 \cdot 1 - 1 \cdot 0 - 1 \cdot 1 = 0$
    - $y_{\text{pred}} =  \overline{w}^T \cdot f(\overline{x}^{[3]}) > 0 = -1$
    - $y_{\text{pred}} = y^{[3]}$, so $\overline{w}_{\text{new}} = \overline{w} = [1, 1, -1, -1]$

**Epoch 3:**
1. First example: $f(\overline{x}^{[1]}) = [1, 1, 0, 0]$, $y^{[1]} = +1$
    - $\overline{w} = [1, 1, -1, -1]$
    - $ \overline{w}^T \cdot f(\overline{x}^{[1]}) = 1 \cdot 1 + 1 \cdot 1 - 1 \cdot 0 - 1 \cdot 0 = 2$
    - $y_{\text{pred}} =  \overline{w}^T \cdot f(\overline{x}^{[1]}) > 0 = +1$
    - $y_{\text{pred}} = y^{[1]}$, so $\overline{w}_{\text{new}} = \overline{w} = [1, 1, -1, -1]$


At this point all examples are correctly classified, so the algorithm has converged, and we can stop.

If they hadn't converged, you would continue to the next epoch, and repeat until the weights no longer change, or until a specified maximum number of epochs is reached. The exact convergence depends on the data and the update rules of your specific algorithm.

### Example 2: Non-Converging
Let's look at an example where we have the following examples:

| movie review | y | $f(\overline{x})$: gbn |
|--------------|---|------------------------|
| good         | +1| 100                    |
| bad          | -1| 010                    |
| not good     | -1| 101                    |
| not bad      | +1| 011                    |

If we drew this on a 3-dimensional graph, we would see that the way the data is laid out, there is no way to draw a line to separate the positive and negative values. 

Therefore, the data is not separable. Perceptron is guaranteed to find a solution or a classification boundary that separates the positive and negative, if one is possible. But in this case, there is no way to do it. 

This illustrates a fundamental issue with the perceptron and unigram features is that they can't model these interactions between words that are quite important.

So, we can expand our feature space to add bi-grams. If we do this, we'll be introducing a 5-dimensional space which will then be separable and we can classify the data. 

By changing the underlying feature set we can go from something that doesn't work to something that does work. We'll explore this more in feature design. 



## Partial derivative explained

From the previous lecture, we said that for stochastic gradient descent, we iterate $t$ times and update our weights with each iteration using the following formula:

$$ \overline{w} \leftarrow \overline{w} - \alpha \frac{\partial}{\partial \overline{w}} \text{loss}(\overline{x}^{[j]}, y^{[j]}, \overline{w}) $$

Let's dive into the calculus to derive the gradient for the perceptron loss function.

Given the perceptron loss function for a misclassified example:

$$ L(w, x, y) = -y \cdot (w \cdot f(x)) $$

Because x and y are constants as we're iterating through, we can just rewrite the function in terms of $w$ like this. 

$$ L(w) = -y \cdot (w \cdot f(x)) $$

To calculate the derivative of this loss with respect to the weights $ w $, we can write out the dot product explicitly and then take the derivative:

1. **Expressing the Loss Function**
   The dot product $ w \cdot f(x) $ can be expressed as a sum:

   $ w \cdot f(x) = \sum_{i=1}^n w_i \cdot f(x)_i $

   where $ n $ is the number of features, $ w_i $ is the i-th element of the weight vector, and $ f(x)_i $ is the i-th element of the feature vector.

2. **Substituting into the Loss Function**
   Substituting this into our loss function, we have:

   $ L(w) = -y \cdot \sum_{i=1}^n w_i \cdot f(x)_i $

3. **Taking the Partial Derivative with Respect to $ w_i $**
   Now, we'll take the partial derivative of the loss function with respect to each weight $ w_i $:

   $ \frac{\partial L}{\partial w_i} = -y \cdot f(x)_i $

   This is because the only term in the sum that depends on $ w_i $ is the term where the sum index equals $ i $, and the derivative of $ w_i \cdot f(x)_i $ with respect to $ w_i $ is simply $ f(x)_i $.

4. **Forming the Gradient Vector**
   The gradient of the loss function with respect to the weight vector $ w $ is the vector of these partial derivatives:

   $ \frac{dL}{dw} = [-y \cdot f(x)_1, -y \cdot f(x)_2, \ldots, -y \cdot f(x)_n] $

**Summary**

The gradient of the perceptron loss function with respect to the weights is a vector pointing in the direction of the feature vector $ f(x) $, scaled by the negative true label $ y $. This leads directly to the perceptron update rules, where the weights are adjusted in the direction that reduces the loss for misclassified examples.
