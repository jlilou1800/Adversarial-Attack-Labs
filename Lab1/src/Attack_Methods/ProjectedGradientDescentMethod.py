import numpy as np

from Lab1.src.Classifiers import SimpleNeuralNetwork


class ProjectedGradientDescentMethod:
    """
    Generates adversarial examples using the Projected Gradient Descent Method (PGD).

    Args:
        x (numpy.ndarray): Input data.
        y (numpy.ndarray): True labels.
        epsilon (float): Perturbation magnitude.
        alpha (float): Step size.
        iterations (int): Number of iterations.
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

    def generate_adversarial_set(self, x_test, y_test, epsilon, alpha=0.1, iterations=10):
        x_adv = []
        for i in range(len(x_test)):
            x_tmp = self.generate(x_test[i], y_test[i], epsilon, alpha, iterations)
            x_adv.append(x_tmp)
        return np.array(x_adv)

    def generate(self, x, y, epsilon, alpha, iterations):
        x_adv = x.copy()

        for _ in range(iterations):
            # Compute the gradient of the loss function with respect to the input
            gradient = self.calculate_gradient(x_adv, y)

            # Compute the sign of the gradient
            sign = np.sign(gradient)

            # Create the adversarial example by adding the sign of the gradient multiplied by alpha to the original image
            x_adv = x_adv + alpha * sign

            # Clip pixel values to the valid range [0, 255]
            x_adv = np.clip(x_adv, 0, 255)

            # Project the perturbed image onto the epsilon ball around the original image
            x_adv = np.clip(x_adv, x - epsilon, x + epsilon)

        return x_adv

    def calculate_gradient(self, x, y):
        # Flatten the input to match the model input shape
        x = x.reshape(-1)

        # Compute model predictions
        predictions = self.model.predict([x])

        # Compute the cross-entropy loss
        loss = self.cross_entropy_loss(y, predictions)

        # Compute the gradient of the loss function with respect to the input
        gradients = self.model.gradient(loss, x)
        d_weights_input_hidden, d_bias_hidden, d_weights_hidden_output, d_bias_output, d_input = gradients

        # Flatten the gradients for input weights to match the input dimension
        gradient = d_input.flatten()

        return gradient

    def cross_entropy_loss(self, y_true, predictions):
        # Convert y_true to index (0-25)
        y_true_index = int(y_true) - ord('a')

        # Create a one-hot encoded array of the same size as the number of classes (26 letters)
        y_true_one_hot = np.zeros(26)
        y_true_one_hot[y_true_index] = 1

        # Prevent log(0) and ensure predictions are valid probabilities
        epsilon = 1e-3  # Small number to avoid log(0)
        predictions = np.clip(predictions, epsilon, 1 - epsilon)

        # Compute cross-entropy loss
        loss = -np.sum(y_true_one_hot * np.log(predictions))

        return loss
