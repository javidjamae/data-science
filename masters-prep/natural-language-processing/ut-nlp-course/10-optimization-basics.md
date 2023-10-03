# Optimization Basics

These are my notes from [this lecture](https://www.youtube.com/watch?v=65ui-GdtY0Q&list=PLofp2YXfp7TZZ5c7HEChs0_wfEfewLDs7&index=10) that is part of [this course](https://www.cs.utexas.edu/~gdurrett/courses/online-course/materials.html).

## Lecture Notes

For optimization we have a loss function and we are finding a set of weights $\overline{w}$ that we're using to minimize that loss. This can be thought of as a search over the space of parameters $\overline{w}$.

The loss is:
$$ \mathscr{L}( ( \overline{x}^{[i]}, y^{[i]}), \overline{w} )$$

Where:
* $(\overline{x}^{[i]}, y^{[i]})$ is a training set that is indexed by $i$ and goes from $1$ to $D$ examples
* $\overline{w}$ is the set of weights

We can think of this loss function as a linear sum over the training examples:

$$ \sum_{i=1}^{D} \mathscr{L}(\overline{x}^{[i]}, y^{[i]}, \overline{w}) $$

This is a function of $\overline{w}$, and the training data is fixed. We change $\overline{w}$ to find a good value for the loss function.

With Stochastic Gradient Descent, we repeatedly pick an example $i$ and apply the update. Because the training data is fixed, we'll just simplify the notation for the loss function to take $i$ as an argument.

$$ \overline{w} \leftarrow \overline{w} - \alpha \frac{\partial}{\partial \overline{w}} \mathscr{L}(i, \overline{w}) $$

Where:
* $\mathscr{L}(i, \overline{w})$ is the loss of the $i$th example
* $\alpha$ is the step size (or learning rate)
* Since we're minimizing the loss, we subtract the gradient from the weight

### Why Step Size Matters

Why does step size matter?

Suppose we have a specific case where the loss function looks like this:

$$\mathscr{L}(i, \overline{w}) = w^2 $$

and suppose that there is only one feature, where the weight vector is just a single coordinate:

$$\overline{w} = [ w ]$$

We can calculate that the gradient of the loss $w^2$ with respect to $w$ is:

$$\frac{\partial}{\partial \overline{w}} \mathscr{L} = 2w$$

To visualize this, consider a graph where the x-axis is the weight $w$ and the y-axis is the loss $w^2$. We know that $w^2$ is a quadratic function (u-shaped) which intercepts on the y-axis at $0$ when $w=0$. So we intuitively know that the minimum loss would one where the weight is $0$ and the loss is $0$.

So now we can rewrite our weight update formula as:

$$ \overline{w} \leftarrow \overline{w} - \alpha \cdot 2w $$

Now, let's assume that our initial starting point has $w = -1$ and we select $\alpha = 1$.

If $\alpha = 1$, and $w = -1$, then we update the weight as $w = -1 \cdot 2( -1 ) = 1$

So, thinking back to our graph where the x-axis is $w$ and the y-axis is loss, we started with a $(\text{weight}, \text{loss})$ coordinate of $(-1, 1)$ and the gradient told us to go towards the origin. But, because of our selection of $\alpha$ we overshot the origin and went to $(1, 1)$.

If we keep $\alpha = 1$ then the algorithm will just oscillate and it will never converge to the true minimum loss, where $w = 0$.

If we select $\alpha = 0.5$ then it will converge. So, the correct step size allows us to get to the minimum. So the step-size is important because it can make a difference between converging and not converging, given the same algorithm. 

### Choosing Step Size

How do we choose step size when doing SGD?

**Try out different sizes** 

Try a range of different orders of magnitude like $1e^-1$, $1e^-2$, $1e^-3$, etc. If you don't know too much about the function you're optimizing, you're not sure what scale you're even searching for. With a lot of neural models, we may want to try $1e^-4$ or $1e^-5$ depending on how big the model is.

**Start large and go small**

For example, $\frac{1}{t}$ or $\frac{1}{\sqrt{t}}$ for epoch $t$. This could be considered a "fixed schedule". But you could also decrease step size when performance stagnates on held-out data. That means when you're no longer making progress on your validation (i.e. development set) you can turn down your step size. 

**Newton's Method**

This is the smarter approach.

$$ \overline{w} \leftarrow \overline{w} - \left(\frac{\partial^2}{\partial \overline{w}^2} \mathscr{L}\right)^{-1} \frac{\partial}{\partial \overline{w}} \mathscr{L}$$

This is called the inverse-Hessian. We use the curvature of the objective function to figure out what the right step size is. It's like a second-order Taylor approximation. For example, if you're optimizing a quadratic, it will take your gradient, which is locally going uphill, it uses the second derivative to decide that it can immediately jump to the optimum. 

The downside is that this is very expensive to compute. It's quadratic to the number of features, so for neural and linear models with 100's or 1000's of features it becomes infeasible. 

**Adagrad, Adadelta, Adam**

These are adaptive methods that are motivated as approximations to the inverse Hessian, but they're linear in the number of features. 

These techniques are useful for deep-learning, and we'll revisit Adam when we get to deep learning and will discuss its hyperparameters there.

### Regularization

We don't really use this. In classical statistics, we would say that fully optimizing the value of the loss is bad. There is a bias/vairance trade off, so we don't want to fully optimize to reduce the variance of our estimator and do better on new data.

The idea of regularization is useful, but we're not going to typically add in regularization to our objective. Instead we'll benefit from:
* early stopping (not running as many iterations)
* the fact that our optimization is not going to be perfect at optimizing a given function
* other adhoc tricks like dropout that will give us the same benefits as regularization of not overfitting the data, without explicitly incorporating it

The main thing for the first part of this course is the step size and knowledge that some of these techniques exist, then we'll revisit some of these concepts when we get to optimization in deep learning, later in the course. 

## Further notes / research

Here are some more notes on topics not covered in detail in the lecture notes.

### Regularization

Regularization is a commonly used technique in machine learning, including in natural language processing (NLP). It is used to prevent overfitting, especially when working with models that have a large number of parameters.

Overfitting occurs when a model learns the noise in the training data, hindering its ability to generalize to unseen data. 

By adding a penalty term to the loss function, regularization constrains the complexity of the model and discourages it from fitting the noise. 

The main types of regularization are L1 (Lasso), L2 (Ridge), and Elastic Net, each imposing different kinds of constraints on the weights.

#### Why We Aren't Using Regularization

While regularization is a pervasive concept in machine learning and often applied in NLP, its omission from this course or discussion doesn't necessarily mean it's not used or valuable in the field. It might simply reflect the specific focus, assumptions, or preferences of the instructor or curriculum.

In many NLP tasks, especially when dealing with high-dimensional data and deep learning models, overfitting may not be the main concern. The following reasons could explain why regularization might not be essential in this course:

1. **Large Datasets:** NLP often involves working with substantial amounts of data. With more data, the model has a better chance of capturing the underlying patterns without fitting to the noise, reducing the need for regularization.

2. **Early Stopping:** By monitoring the performance on a validation set and stopping training when performance plateaus, overfitting can be mitigated without explicitly incorporating regularization.

3. **Dropout and Other Techniques:** In deep learning for NLP, techniques like dropout can be employed. Dropout randomly sets a fraction of input units to 0 during training, which can prevent complex co-adaptations and function as a form of regularization.

4. **Inherent Complexity:** Some NLP models, particularly deep learning architectures, may already contain mechanisms to control overfitting. Fine-tuning hyperparameters and architectural decisions can often be more critical than adding explicit regularization terms.

5. **Course Focus:** Depending on the goals and topics covered in this course, regularization might not be the focal point. If the emphasis is on understanding specific algorithms, optimization methods, or particular applications within NLP, regularization may be beyond the scope or less relevant to the learning objectives.

By considering these factors, it's possible to achieve strong performance in many NLP tasks without relying heavily on traditional regularization methods.

#### L1 Regularization (Lasso)

L1 Regularization, also known as Lasso, adds a penalty term proportional to the absolute values of the weights:

$ \text{Penalty} = \lambda \sum |w_i| $

where $\lambda$ is the regularization strength. It can lead to some weights being exactly zero, effectively leading to feature selection.

#### L2 Regularization (Ridge)

L2 Regularization, or Ridge, adds a penalty term proportional to the squares of the weights:

$ \text{Penalty} = \lambda \sum w_i^2 $

It can help prevent overfitting by constraining the weights but does not generally lead to zero weights.

#### Elastic Net

Elastic Net combines L1 and L2 regularization. It's useful when there are multiple correlated features.

### Newton's Method

#### Intuition
The intuition behind Newton's Method is that it takes into account not only the gradient (first derivative) of the objective function (cost function) but also the curvature (second derivative) at the current solution. This allows the algorithm to take larger steps when the curvature is gentle and smaller steps when the curvature is steep, resulting in faster convergence toward the optimal solution.

However, there are some important considerations with Newton's Method:

- **Computational Cost:** Calculating and inverting the Hessian matrix can be computationally expensive, especially when dealing with high-dimensional problems.
- **Convergence Issues:** Newton's Method might not always converge, especially when the Hessian is ill-conditioned or the initial guess is far from the optimum.
- **Positive Definiteness:** The Hessian matrix must be positive definite for the optimization process to work properly.

Newton's Method can be applied to various optimization problems, including both linear and non-linear cases. In the context of machine learning and optimization, Newton's Method can be very effective for certain types of problems but might not be suitable for all scenarios due to its computational demands and potential convergence issues. In practice, approximate versions like the Quasi-Newton L-BFGS algorithm are often used to mitigate these challenges.

#### Compared to Perceptron
Newton's Method and the Perceptron algorithm serve different purposes and have different underlying principles. The Perceptron algorithm is specifically designed for binary classification tasks and focuses on updating weights based on misclassified instances. It's a simple and fast algorithm that aims to find a linear decision boundary that separates two classes. 

On the other hand, Newton's Method is a more general optimization algorithm that can be used to find the minimum (or maximum) of any differentiable function, not just for classification tasks. While Newton's Method can theoretically be applied to optimization problems in machine learning, it's not commonly used for training classifiers due to its computational complexity, convergence issues, and suitability for high-dimensional problems.

### Adaptive Methods

Here is a summary of some of the Adaptive Methods for selecting the $\alpha$ hyperparameter.

#### Adagrad

Adagrad adjusts the learning rate for each parameter based on the historical gradient information. It makes the learning rate decrease quickly for frequently occurring features:

$ g_{t,i} = \nabla_{\theta_i} \mathscr{L}(\theta_i) $
$ s_{t,i} = s_{t-1,i} + g_{t,i}^2 $
$ \theta_{t+1,i} = \theta_{t,i} - \frac{\alpha}{\sqrt{s_{t,i} + \epsilon}} \cdot g_{t,i} $

where $\theta$ are the parameters, $\mathscr{L}$ is the loss function, and $\epsilon$ is a smoothing term.

#### Adadelta

Adadelta is an extension of Adagrad that reduces its aggressive decaying learning rate. Instead of accumulating all past squared gradients, Adadelta restricts the window of accumulated past gradients to a fixed size.

#### Adam

Adam (Adaptive Moment Estimation) combines ideas from both Adagrad and RMSprop. It calculates an exponential moving average of the gradients and the squared gradients:

$ m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t $
$ v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 $
$ \theta_{t+1} = \theta_t - \alpha \cdot \frac{\hat{m_t}}{\sqrt{\hat{v_t}} + \epsilon} $

where $ m_t $ and $ v_t $ are estimates of the first and second moments (mean and uncentered variance) of the gradients.
