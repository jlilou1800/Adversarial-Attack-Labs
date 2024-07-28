import numpy as np

from Lab1.src.Classifiers import SimpleNeuralNetwork


class CarliniWagnerMethod:
    """
    Generates adversarial examples using the Carlini Wagner Method.

    Args:
        x (numpy.ndarray): Input data.
        y (numpy.ndarray): True labels.
        epsilon (float): Perturbation magnitude.
        confidence (float): Confidence level for the attack.
        targeted (bool, optional): Whether to perform targeted attack. Defaults to False.

    Returns:
        numpy.ndarray: Adversarial examples.
    """
    def __init__(self, width, height):
        self.img_width = width
        self.img_height = height

        #TODO Generer le simple neural network hors du cette classe
        input_size = width * height  # Nombre total de caractéristiques par exemple
        hidden_size = 2*26
        output_size = 26  # Nombre total de classes (lettres de l'alphabet)
        self.model = SimpleNeuralNetwork.SimpleNeuralNetwork(input_size, hidden_size, output_size)
        self.weights_input_hidden = 0
        self.bias_hidden = 0
        self.weights_hidden_output = 0
        self.bias_output = 0

    def generate_adversarial_set(self, x_test, y_test, epsilon, confidence=10, targeted=False):
        x_adv = []
        for i in range(len(x_test)):
            x_tmp = self.generate(x_test[i], y_test[i], epsilon, confidence, targeted)
            x_adv.append(x_tmp.flatten())  # Flatten the image to 1D
        return np.array(x_adv)

    def generate(self, x, y, epsilon, confidence, targeted=False):
        x_adv = x.copy()

        # Perform the Carlini-Wagner attack
        # Implementation details can vary, and the following is a simplified version for demonstration purposes

        # Binary search for the optimal perturbation magnitude
        # Initialize lower and upper bounds
        lower_bound = 0
        upper_bound = 1e10

        # Set the binary search threshold
        threshold = 0.01

        # Perform binary search until the desired confidence level is achieved
        while upper_bound - lower_bound > threshold:
            # Choose a candidate perturbation magnitude
            perturbation_magnitude = (lower_bound + upper_bound) / 2.0

            # Generate adversarial examples with the chosen perturbation magnitude
            x_adv = self.generate_adversarial_examples(x, y, epsilon, perturbation_magnitude, targeted)

            # Check the confidence level of the adversarial examples
            # If the confidence level exceeds the desired threshold, reduce the upper bound
            # Otherwise, increase the lower bound
            if self.check_confidence(x_adv, y, targeted) >= confidence:
                upper_bound = perturbation_magnitude
            else:
                lower_bound = perturbation_magnitude

        return x_adv

    def generate_adversarial_examples(self, x, y, epsilon, perturbation_magnitude, targeted):
        # Compute the gradient of the loss function with respect to the input
        x = x.reshape(-1)
        gradient = self.calculate_gradient(x, y, targeted)

        # Create the adversarial example by adding the perturbation to the original image
        perturbation = epsilon * np.sign(gradient)
        x_adv = x + perturbation * perturbation_magnitude

        # Clip pixel values to the valid range [0, 255]
        x_adv = np.clip(x_adv, 0, 255)

        return x_adv.reshape(self.img_width, self.img_height)

    def calculate_gradient(self, x, y, targeted):
        # Flatten the input to match the model input shape
        x = x.reshape(-1)

        # Compute model predictions
        predictions = self.model.predict([x])

        # If targeted attack, use target class, else use true class
        target_class = np.argmax(predictions) if not targeted else y

        # Compute the gradient of the loss function with respect to the target class
        gradient = self.model.gradient_wrt_target_class(x, target_class)

        return gradient

    def check_confidence(self, x_adv, y, targeted):
        # Compute model predictions
        predictions = self.model.predict([x_adv.reshape(-1)])

        # If targeted attack, check the confidence for the target class
        if targeted:
            confidence = predictions[y]
        else:
            # If untargeted attack, check the confidence for the least likely class
            confidence = 1 - np.max(predictions)

        return confidence
