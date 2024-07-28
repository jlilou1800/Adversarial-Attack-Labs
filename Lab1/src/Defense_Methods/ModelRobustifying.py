import copy

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator


class ModelRobustifying:
    """
    Defense method that robustifies the model against adversarial attacks.
    """

    def __init__(self, model):
        """
        Initializes the ModelRobustifying defense.

        Args:
            model: The model to be defended.
        """
        self.model = copy.deepcopy(model)

    def defend(self, x_train, y_train, epsilon, nb_iter):
        """
        Enhances model robustness through robustifying the model.

        Args:
            x_train: Original training data.
            y_train: Corresponding labels for the training data.
            epsilon: Perturbation magnitude for generating adversarial examples.
            nb_iter: Number of iterations for model robustifying.
        """
        for _ in range(nb_iter):
            adv_x = generate_adversarial_examples(self.model, x_train, y_train, epsilon)
            self.model.classifier.fit(adv_x, y_train)

        return self.model

def generate_adversarial_examples(model, x_train, y_train, epsilon):
    """
    Generates adversarial examples for the given model and training data.

    Args:
        model: The model to attack.
        x_train: Original training data.
        y_train: Corresponding labels for the training data.
        epsilon: Perturbation magnitude.

    Returns:
        Adversarial examples.
    """
    onehot_encoder = OneHotEncoder(sparse=False)
    y_train_onehot = onehot_encoder.fit_transform(y_train.reshape(-1, 1))

    predictions = model.predict_proba(x_train)
    loss_gradient = predictions - y_train_onehot

    grad = np.sign(loss_gradient)
    grad = np.mean(grad, axis=1, keepdims=True)  # Ensure the gradient matches the input dimensions

    adv_x = x_train + epsilon * grad
    adv_x = np.clip(adv_x, 0, 1)  # Ensure values are within valid range

    return adv_x