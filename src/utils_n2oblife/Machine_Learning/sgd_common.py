import numpy as np

def error(
        Xest: float | np.ndarray, 
        target: float | np.ndarray
    ) -> float | np.ndarray:
    """
    Calculate the error between the estimated and target values.

    Args:
        Xest (float | np.ndarray): The estimated value(s).
        target (float | np.ndarray): The target value(s).

    Returns:
        float | np.ndarray: The difference between the estimated and target values.
    """
    return np.abs(Xest - target)

def loss(
        Xest: list | np.ndarray, 
        target: list | np.ndarray
    ) -> float | np.ndarray:
    """
    Calculate the loss (sum of squared errors) between the estimated and target values.

    Args:
        Xest (list | np.ndarray): The estimated value(s).
        target (list | np.ndarray): The target value(s).

    Returns:
        float | np.ndarray: The sum of squared errors between the estimated and target values.
    """
    if isinstance(Xest, np.ndarray):
        return np.sum(error(Xest, target)**2)
    else:
        return sum([error(Xest[i], target[i])**2 for i in range(len(Xest))])

def sgd_step(
        coeff: float | np.ndarray, 
        lr: float | np.ndarray, 
        delta: float | np.ndarray, 
        bias: float | np.ndarray = 0
    ) -> float | np.ndarray:
    """
    Perform a single step of stochastic gradient descent (SGD) to update a coefficient.

    Args:
        coeff (float | np.ndarray): The current value of the coefficient.
        lr (float | np.ndarray): The learning rate for the update step.
        delta (float | np.ndarray): The gradient of the loss function with respect to the coefficient.
        bias (float | np.ndarray, optional): An optional bias term to be added to the update. Defaults to 0.

    Returns:
        float | np.ndarray: The updated value of the coefficient after the SGD step.
    """
    return coeff - lr * delta + bias

def Xest(
        g: float | np.ndarray, 
        y: float | np.ndarray, 
        o: float | np.ndarray, 
        b: float | np.ndarray = 0.
    ) -> float | np.ndarray:
    """
    Estimate a value using a linear combination of inputs and an optional bias.

    Args:
        g (float | np.ndarray): The first input value or coefficient.
        y (float | np.ndarray): The second input value or coefficient.
        o (float | np.ndarray): The third input value or coefficient.
        b (float | np.ndarray, optional): An optional bias term. Defaults to 0.

    Returns:
        float | np.ndarray: The estimated value resulting from the linear combination of the inputs and the bias.
    """
    return g * y + o + b