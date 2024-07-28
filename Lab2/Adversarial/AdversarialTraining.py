from Lab2.Adversarial.AdversarialExampleGenerator import AdversarialExampleGenerator


class AdversarialTraining:
    """
    Class for adversarial training defense method.
    """

    def __init__(self, model, input_size):
        """
        Initializes the AdversarialTraining class.

        Args:
            model (tf.keras.Model): The model to be defended.
            input_size (int): The size of the input features.
        """
        self.model = model
        self.input_size = input_size

    def adversarial_training(self, attack_function, x_train, y_train, epsilon, epochs):
        """
        Enhances model robustness through adversarial training.

        Args:
            attack_function (callable): Attack method to generate adversarial examples.
            x_train (numpy.ndarray): Original training data.
            y_train (numpy.ndarray): Corresponding labels for the training data.
            epsilon (float): Perturbation magnitude for generating adversarial examples.
            epochs (int): Number of epochs for adversarial training.
        """
        for _ in range(epochs):
            adv_x_train = attack_function(AdversarialExampleGenerator(self.model, self.input_size), x_train, epsilon)
            self.model.fit(adv_x_train, y_train, epochs=1, batch_size=256, verbose=0)
