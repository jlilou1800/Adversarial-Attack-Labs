import copy

import numpy as np


class AdversarialTraining:
    """
    Defense method that enhances model robustness by training on both original and adversarial examples.
    """

    def __init__(self, model):
        """
        Initializes the AdversarialTraining defense.

        Args:
            model: The model to be defended.
        """
        self.model = copy.deepcopy(model)

    def defend(self, x_train, y_train, epsilon, attack_method, nb_iter=5):
        """
        Enhances model robustness through adversarial training.

        Args:
            x_train: Original training data.
            y_train: Corresponding labels for the training data.
            epsilon: Perturbation magnitude for generating adversarial examples.
            nb_iter: Number of iterations for adversarial training.
        """
        for _ in range(nb_iter):
            adv_x = attack_method.generate_adversarial_set(x_train, y_train, epsilon)
            x_combined = np.concatenate([x_train, adv_x])
            y_combined = np.concatenate([y_train, y_train])

            # Flatten the input data to match the model input shape
            x_combined = x_combined.reshape((x_combined.shape[0], -1))

            # Train the model on the combined dataset
            self.model.classifier.fit(x_combined, y_combined)

        return self.model