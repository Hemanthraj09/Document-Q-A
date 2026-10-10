"""A small study-notes document with a table of contents, running headers, page numbers and clear dependencies."""

HEADER = "ML Study Notes - Unit 1"

_BODIES = [
    ("Machine Learning Foundations",
     "Contents\n"
     "Loss functions and how models are scored (page 2)\n"
     "Gradient Descent and the learning rate (page 3)\n"
     "Neural Networks and their layers (page 4)\n"
     "Backpropagation through the network (page 5)\n"
     "Overfitting and why models memorize (page 6)\n"
     "Regularization and its penalty terms (page 7)"),
    ("Loss Function",
     "A loss function is a measure of how wrong a model's predictions are on the training data.\n"
     "The loss function compares each prediction with the correct target value.\n"
     "Lower values of the loss function mean the model fits the training data better.\n"
     "Every training method in these notes tries to reduce the loss function in some way."),
    ("Gradient Descent",
     "Gradient descent is an optimization algorithm that minimizes a loss function by following its gradient.\n"
     "The learning rate controls how large each step of gradient descent is.\n"
     "A learning rate that is too large makes gradient descent overshoot the minimum.\n"
     "For example, a learning rate of 0.01 is a common starting point for gradient descent."),
    ("Neural Networks",
     "A neural network is a model made of layers of connected neurons that transform their inputs.\n"
     "Neural networks are trained with gradient descent on a loss function.\n"
     "Each neural network layer applies weights and a nonlinear activation to its inputs.\n"
     "A data pipeline feeds batches of examples into the neural network during training."),
    ("Backpropagation",
     "Backpropagation is an algorithm that computes the gradient of the loss function for every weight in a neural network.\n"
     "Backpropagation relies on gradient descent to turn those gradients into weight updates.\n"
     "Without backpropagation, training a deep neural network with gradient descent would be far too slow.\n"
     "See Figure 2 for how backpropagation moves errors backwards through each layer."),
    ("Overfitting",
     "Overfitting is when a neural network memorizes the training data instead of learning general patterns.\n"
     "An overfitted neural network scores well on training data but badly on new data.\n"
     "Regularization reduces overfitting, as the next section explains in detail.\n"
     "For example, a very large neural network trained for too long on a small data pipeline often shows overfitting."),
    ("Regularization",
     "Regularization is a technique that adds a penalty to the loss function to reduce overfitting.\n"
     "The most common regularization penalty grows with the size of the neural network weights.\n"
     "Stronger regularization lowers overfitting but can make the loss function harder to minimize.\n"
     "Gradient descent then minimizes the loss function together with the regularization penalty."),
]


def pages() -> list[str]:
    return [f"{HEADER}\n{title}\n{body}\n{n}" for n, (title, body) in enumerate(_BODIES, start=1)]


def headings() -> list[dict]:
    return [{"text": title, "level": 1, "page": n} for n, (title, _) in enumerate(_BODIES, start=1)]
