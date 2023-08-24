# Logistic Regression

These are my notes from [this lecture](https://www.youtube.com/watch?v=0naHFT07ja8&list=PLofp2YXfp7TZZ5c7HEChs0_wfEfewLDs7&index=8) that is part of [this course](https://www.cs.utexas.edu/~gdurrett/courses/online-course/materials.html). 

## Lecture Notes

[TODO]

## Compared to Perceptron

### Logistic Regression
1. **Model Functionality**: Logistic regression models the probability that the dependent variable belongs to a particular category. It outputs a probability score between 0 and 1.
2. **Activation Function**: It uses the logistic (or sigmoid) function to squash the output between 0 and 1.
3. **Cost Function**: Typically uses log loss or cross-entropy loss, which is differentiable. This leads to a smooth loss landscape that makes optimization easier.
4. **Learning Algorithm**: Gradient-based optimization algorithms like gradient descent are usually employed.
5. **Probabilistic Interpretation**: Outputs can be interpreted as probabilities, providing not just a class label but also information on how confident the model is about that prediction.
6. **Applicability**: Suitable for problems where you need the probabilities or when the decision boundary needs to be more flexible (it can learn nonlinear decision boundaries with added features).

### Perceptron Algorithm
1. **Model Functionality**: The perceptron algorithm aims to find a hyperplane that separates the classes. It makes hard binary decisions, i.e., outputs are 0 or 1.
2. **Activation Function**: Typically uses a step function to produce binary outputs.
3. **Cost Function**: Uses a loss function that is not differentiable everywhere, leading to a more complex optimization landscape.
4. **Learning Algorithm**: Works by making updates in response to misclassifications, using a simple rule that doesn’t require differentiation.
5. **Probabilistic Interpretation**: No probabilistic interpretation of the outputs, just class labels.
6. **Applicability**: Works best when the data is linearly separable or nearly so. It might fail to converge if the classes are not perfectly separable.

### Comparison Summary
- **Use Logistic Regression when**:
  - You want probability estimates.
  - Your data might not be linearly separable.
  - You have a large and complex dataset (since logistic regression can be more robust and handle noise better).

- **Use Perceptron when**:
  - You only need hard binary classifications.
  - You have linearly separable data or nearly so.
  - You prefer a simpler, faster algorithm with fewer assumptions.
