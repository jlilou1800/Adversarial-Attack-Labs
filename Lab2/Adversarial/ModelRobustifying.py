from Lab2.Adversarial.AdversarialExampleGenerator import AdversarialExampleGenerator


class ModelRobustifying:
    """
    Class for model robustifying defense method.
    """

    def __init__(self, model, input_size):
        """
        Initializes the ModelRobustifying class.

        Args:
            model (tf.keras.Model): The model to be defended.
            input_size (int): The size of the input features.
        """
        self.model = model
        self.input_size = input_size

    def model_robustifying(self, x_train, y_train, epsilon, iterations):
        """
        Enhances model robustness through model robustifying.

        Args:
            x_train (numpy.ndarray): Original training data.
            y_train (numpy.ndarray): Corresponding labels for the training data.
            epsilon (float): Perturbation magnitude for generating adversarial examples.
            iterations (int): Number of iterations for model robustifying.
        """
        for _ in range(iterations):
            adv_x_train = AdversarialExampleGenerator(self.model, self.input_size).generate_fgsm_attack(x_train, epsilon)
            self.model.fit(adv_x_train, y_train, epochs=1, batch_size=256, verbose=0)