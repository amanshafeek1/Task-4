# Understanding the Sigmoid Function in Logistic Regression

The **Sigmoid Function** (also known as the Logistic Function) is the core component of Logistic Regression. It maps any real-valued number into a value between 0 and 1, making it ideal for predicting probabilities.

## The Formula

The mathematical representation of the Sigmoid function is:

$$ \sigma(z) = \frac{1}{1 + e^{-z}} $$

Where:
- $z$ is the output of the linear combination of inputs ($z = w \cdot x + b$).
- $e$ is Euler's number (~2.718).

## Why Use It?

1.  **Probability Output**: Since the output is always between $[0, 1]$, we can interpret it as the probability of an instance belonging to a certain class.
2.  **Differentiability**: The function is smooth and differentiable everywhere, which is crucial for optimization algorithms like Gradient Descent.
3.  **Decision Threshold**: We can set a threshold (commonly 0.5) to decide the final class:
    - If $\sigma(z) \geq 0.5$, predict **Class 1** (e.g., Benign).
    - If $\sigma(z) < 0.5$, predict **Class 0** (e.g., Malignant).

## Visual Representation

The curve looks like an "S" shape:
- As $z \to \infty$, $\sigma(z) \to 1$.
- As $z \to -\infty$, $\sigma(z) \to 0$.
- When $z = 0$, $\sigma(z) = 0.5$.

In our breast cancer classifier, the model learns the weights $w$ such that "healthy" features push the output toward 1 and "dangerous" features push it toward 0.
