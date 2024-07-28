import numpy as np

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(hidden_size, input_size)
        self.bias_hidden = np.zeros((hidden_size, 1))
        self.weights_hidden_output = np.random.randn(output_size, hidden_size)
        self.bias_output = np.zeros((output_size, 1))

    def softmax(self, x):
        """Compute softmax values for each set of scores in x."""
        e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return e_x / e_x.sum(axis=0, keepdims=True)

    def sigmoid(self, x):
        """ Stable sigmoid activation function to avoid overflow. """
        # Avoid overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, output):
        """ Derivative of the sigmoid function, `output` is the sigmoid function result. """
        return output * (1 - output)

    def gradient(self, loss, x):
        """Calculate gradients of the loss with respect to the input x."""
        # First, perform a forward pass
        self.forward(x)

        # Assume loss is a one-hot encoded true label vector
        d_loss_d_output = self.predicted_probs - loss

        # Backpropagate this gradient through the output to hidden layer
        d_output_d_hidden = np.dot(self.weights_hidden_output.T, d_loss_d_output)

        # Sigmoid derivative for the hidden layer output
        sigmoid_deriv = self.sigmoid_derivative(self.hidden_output)

        # Calculate the gradient with respect to the hidden input
        d_hidden_d_input = d_output_d_hidden * sigmoid_deriv

        # Calculate gradients for weights and biases
        d_weights_hidden_output = np.dot(d_loss_d_output, self.hidden_output.T)
        d_bias_output = np.sum(d_loss_d_output, axis=1, keepdims=True)

        d_weights_input_hidden = np.dot(d_hidden_d_input, self.input_layer.T)
        d_bias_hidden = np.sum(d_hidden_d_input, axis=1, keepdims=True)


        # Calculate the gradient with respect to the input x
        d_input = np.dot(self.weights_input_hidden.T, d_hidden_d_input)

        # Reshape the gradient to match the input dimensions
        d_input = d_input.reshape(x.shape)

        return d_weights_input_hidden, d_bias_hidden, d_weights_hidden_output, d_bias_output, d_input

    def forward(self, x):
        """Perform a forward pass of the network."""
        # Convert input x to a NumPy array if it's not already, and ensure it is a column vector
        if isinstance(x, list):
            x = np.array(x, dtype=np.float32)  # Convert list to numpy array if necessary
        x = x.reshape(-1, 1)  # Ensure it is a column vector, assuming x is a flat array originally
        self.input_layer = x
        self.hidden_output = self.sigmoid(np.dot(self.weights_input_hidden, self.input_layer) + self.bias_hidden)
        self.output_logits = np.dot(self.weights_hidden_output, self.hidden_output) + self.bias_output
        self.predicted_probs = self.softmax(self.output_logits)

        return self.predicted_probs

    def predict(self, x):
        """Predict class labels for samples in x."""
        probabilities = self.forward(x)
        return np.argmax(probabilities, axis=0)  # Return the class with the highest probability

    def predict_proba(self, x):
        probabilities = self.forward(x)
        return probabilities.flatten()

    def backward(self, y):
        """Perform a backward pass of the network, updating weights and biases."""
        # Assume 'y' is the index of the true class label
        true_labels = np.eye(self.bias_output.shape[0])[:, y].reshape(-1, 1)  # One-hot encoding of y

        # Use 'self.predicted_probs' which were calculated during the forward pass
        d_output = self.predicted_probs - true_labels

        # Calculate gradients for weights and biases between hidden and output layers
        d_weights_hidden_output = np.dot(d_output, self.hidden_output.T)
        d_bias_output = np.sum(d_output, axis=1, keepdims=True)

        # Backpropagate through the network
        d_hidden_output = np.dot(self.weights_hidden_output.T, d_output)
        d_hidden_input = d_hidden_output * sigmoid_derivative(self.hidden_output)

        # Calculate gradients for weights and biases between input and hidden layers
        d_weights_input_hidden = np.dot(d_hidden_input, self.input_layer.T)
        d_bias_hidden = np.sum(d_hidden_input, axis=1, keepdims=True)

        # Update weights and biases using a simple learning rate
        self.weights_hidden_output -= 0.01 * d_weights_hidden_output
        self.bias_output -= 0.01 * d_bias_output
        self.weights_input_hidden -= 0.01 * d_weights_input_hidden
        self.bias_hidden -= 0.01 * d_bias_hidden

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def gradient_wrt_target_class(self, x, target_class):
        # Perform a forward pass
        self.forward(x)

        # Create a one-hot encoded vector for the target class
        target = np.zeros(self.output_size)
        target[target_class] = 1

        # Calculate the gradient with respect to the target class
        d_loss_d_output = self.predicted_probs - target.reshape(-1, 1)

        # Backpropagate this gradient through the output to hidden layer
        d_output_d_hidden = np.dot(self.weights_hidden_output.T, d_loss_d_output)

        # Sigmoid derivative for the hidden layer output
        sigmoid_deriv = self.sigmoid_derivative(self.hidden_output)

        # Calculate the gradient with respect to the hidden input
        d_hidden_d_input = d_output_d_hidden * sigmoid_deriv

        # Calculate the gradient with respect to the input x
        d_input = np.dot(self.weights_input_hidden.T, d_hidden_d_input)

        # Reshape the gradient to match the input dimensions
        d_input = d_input.reshape(x.shape)

        return d_input

    def gradient_wrt_least_likely_class(self, x, least_likely_class):
        # Perform a forward pass
        self.forward(x)

        # Create a one-hot encoded vector for the least likely class
        target = np.zeros(self.output_size)
        target[least_likely_class] = 1

        # Calculate the gradient with respect to the least likely class
        d_loss_d_output = self.predicted_probs - target.reshape(-1, 1)

        # Backpropagate this gradient through the output to hidden layer
        d_output_d_hidden = np.dot(self.weights_hidden_output.T, d_loss_d_output)

        # Sigmoid derivative for the hidden layer output
        sigmoid_deriv = self.sigmoid_derivative(self.hidden_output)

        # Calculate the gradient with respect to the hidden input
        d_hidden_d_input = d_output_d_hidden * sigmoid_deriv

        # Calculate the gradient with respect to the input x
        d_input = np.dot(self.weights_input_hidden.T, d_hidden_d_input)

        # Reshape the gradient to match the input dimensions
        d_input = d_input.reshape(x.shape)

        return d_input

def softmax(x):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return e_x / e_x.sum(axis=0, keepdims=True)

def sigmoid(x):
    """Sigmoid activation function using a stable approach to avoid overflow and divide by zero errors."""
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)