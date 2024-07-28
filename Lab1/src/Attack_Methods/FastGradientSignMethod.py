import numpy as np
from Lab1.src.Classifiers import SimpleNeuralNetwork


class FastGradientSignMethod:
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

    def train(self, X_train, y_train, epochs, learning_rate):
        for epoch in range(epochs):
            for x, y in zip(X_train, y_train):
                x = x.reshape(-1, 1)
                y = y.reshape(-1, 1)

                # Forward pass
                predictions = self.forward(x)

                # Compute the loss
                loss = self.cross_entropy_loss(y, predictions)

                # Backward pass (compute gradients)
                d_weights_input_hidden, d_bias_hidden, d_weights_hidden_output, d_bias_output = self.gradient(y, x)

                # Update weights and biases
                self.weights_input_hidden -= learning_rate * d_weights_input_hidden
                self.bias_hidden -= learning_rate * d_bias_hidden
                self.weights_hidden_output -= learning_rate * d_weights_hidden_output
                self.bias_output -= learning_rate * d_bias_output

    def generate_adversarial_set(self, x_test, y_test, epsilon):
        x_adv = []
        for i in range(len(x_test)):
            x_tmp = self.generate(x_test[i], y_test[i], epsilon)
            x_adv.append(x_tmp)
        return np.array(x_adv)

    def generate(self, x, y, epsilon):
        # Flatten the 2D array into a 1D array
        x = x.reshape(-1)

        # Compute the gradient of the loss function with respect to the input
        gradient = self.calculate_gradient(x, y)

        # Compute the sign of the gradient
        sign = np.sign(gradient)

        # Create the adversarial example by adding the sign of the gradient multiplied by epsilon to the original image
        x_adv = x + epsilon * sign

        # Clip pixel values to the valid range [0, 1]
        x_adv = np.clip(x_adv, 0, 255)

        return x_adv

    def calculate_gradient(self, x, y):
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
        # Convertir y_true en index (0-25)
        y_true_index = int(y_true) - ord('a')

        # Créer un tableau one-hot encoding de la même taille que le nombre de classes (26 lettres)
        y_true_one_hot = np.zeros(26)
        y_true_one_hot[y_true_index] = 1

        # Prévenir log(0) et s'assurer que les prédictions sont des probabilités valides
        epsilon = 1e-3  # Petit nombre pour éviter log(0)
        predictions = np.clip(predictions, epsilon, 1 - epsilon)

        # Calculer la perte d'entropie croisée
        loss = -np.sum(y_true_one_hot * np.log(predictions))

        return loss
