import numpy as np

from Lab1.src.Classifiers import SimpleNeuralNetwork


class OneStepTargetClassMethod:
    """
    Generates adversarial examples using the One Step Target Class Method.

    Args:
        x (numpy.ndarray): Input data.
        y (numpy.ndarray): True labels.
        epsilon (float): Perturbation magnitude.
        alpha (float): Step size.
        target_class (int): Target class label for targeted attack.

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

    def generate_adversarial_set(self, x_test, y_test, epsilon, alpha=0.1, target_class=120):   # target ord('x')
        x_adv = []
        for i in range(len(x_test)):
            x_tmp = self.generate(x_test[i], y_test[i], epsilon, alpha, target_class)
            x_adv.append(x_tmp)
        return np.array(x_adv)

    def generate(self, x, y, epsilon, alpha, target_class):
        x_adv = x.copy()
        target_class -= ord('a')
        # Compute the gradient of the loss function with respect to the input
        gradient = self.calculate_gradient(x_adv, target_class)

        # Compute the sign of the gradient
        sign = np.sign(gradient)

        # Create the adversarial example by adding the sign of the gradient multiplied by alpha to the original image
        x_adv = x_adv + alpha * sign

        # Clip pixel values to the valid range [0, 255]
        x_adv = np.clip(x_adv, 0, 255)

        # Project the perturbed image onto the epsilon ball around the original image
        x_adv = np.clip(x_adv, x - epsilon, x + epsilon)

        return x_adv

    def calculate_gradient(self, x, target_class):
        # Flatten the input to match the model input shape
        x = x.reshape(-1)

        # Compute the gradient of the loss function with respect to the target class
        gradient = self.model.gradient_wrt_target_class(x, target_class)

        return gradient