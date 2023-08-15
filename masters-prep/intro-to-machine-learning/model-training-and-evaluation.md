# Tuning

## Precision-Recall Curve
Precision-recall curves provide a visual representation of the trade-offs between precision and recall, allowing practitioners to make informed decisions based on the context of the problem at hand.

## ROC / ROC-AUC


# Regularization
Regularization adds a penalty term to the objective function (such as the loss function) that the model is trying to minimize. This penalty discourages the model from fitting the training data too closely, effectively constraining the complexity of the model. There are several types of regularization techniques, and the two most common ones are:

- **L1 Regularization (Lasso)**
   L1 regularization adds the sum of the absolute values of the model parameters (weights) as a penalty term to the loss function. It can result in some of the weights being exactly zero, effectively leading to a sparse representation.

- **L2 Regularization (Ridge)**
   L2 regularization adds the sum of the squares of the model parameters as a penalty term to the loss function. It tends to shrink the weights towards zero but usually doesn't result in exactly zero weights.

- **Elastic Net**
   Elastic Net combines both L1 and L2 regularization, balancing the two based on a hyperparameter.

## How Regularization Works
- Prevents Overfitting: By adding a penalty term, regularization discourages the model from fitting the noise in the training data, aiding in generalization to new data.
- Controls Complexity: Regularization effectively adds a constraint to the optimization process, controlling the complexity of the model.
- Requires Hyperparameter Tuning: The strength of the regularization is controlled by a hyperparameter, often denoted as lambda (λ). Selecting the right value for this hyperparameter is crucial and often requires experimentation and cross-validation.

## Summary
Regularization is a valuable technique in machine learning for controlling the complexity of the model and preventing overfitting. By adding a penalty term to the loss function, it encourages the model to have smaller weights, leading to a more robust model that generalizes well to unseen data.
