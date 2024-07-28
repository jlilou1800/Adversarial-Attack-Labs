import copy

import numpy as np

class DefensiveDistillation:
    """
    Defense method that uses temperature scaling to improve model robustness.
    """

    def __init__(self, model, temperature=5):
        """
        Initializes the DefensiveDistillation defense.

        Args:
            model: The model to be defended.
            temperature: Temperature parameter for softening logits (default=5).
        """
        self.model = copy.deepcopy(model)
        self.temperature = temperature

    def soften_logits(self, logits):
        """
        Applies temperature scaling to logits to soften them.

        Args:
            logits: Logits from the model.

        Returns:
            Softened logits.
        """
        return logits / self.temperature

    def harden_labels(self, softened_logits):
        """
        Converts softened logits into hard labels.

        Args:
            softened_logits: Softened logits from the model.

        Returns:
            Hard labels.
        """
        return np.argmax(softened_logits, axis=1)

    def defend(self, x_train, y_train, nb_iter=5):
        """
        Enhances model robustness through defensive distillation.

        Args:
            x_train: Original training data.
            y_train: Corresponding labels for the training data.
            nb_iter: Number of iterations for defensive distillation.
        """
        for _ in range(nb_iter):
            # Flatten the input data to match the model input shape
            x_train_flat = x_train.reshape((x_train.shape[0], -1))

            # Predict logits and apply temperature scaling
            logits = self.model.predict_proba(x_train_flat)
            softened_logits = self.soften_logits(logits)

            # Convert softened logits to hard labels
            hardened_labels = self.harden_labels(softened_logits)

            # Train the model on the hardened labels
            self.model.classifier.fit(x_train_flat, hardened_labels)

        return self.model