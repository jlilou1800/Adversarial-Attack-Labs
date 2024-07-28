import copy

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam


class InputReconstruction:
    """
    Defense method that reconstructs input data to mitigate adversarial perturbations.
    """

    def __init__(self, model):
        """
        Initializes the InputReconstruction defense.

        Args:
            model: The model to be defended.
        """
        self.model = copy.deepcopy(model)

    def defend(self, x_train, y_train, x_test, nb_iter):
        """
        Enhances model robustness through input reconstruction.

        Args:
            x_train: Original training data.
            y_train: Corresponding labels for the training data.
            x_test: Test data to be reconstructed.
            nb_iter: Number of iterations for input reconstruction.
        """
        autoencoder = train_autoencoder(x_train, nb_iter)
        reconstructed_x_train = autoencoder.predict(x_train)
        reconstructed_x_test = autoencoder.predict(x_test)

        # Train the model on the reconstructed training data
        self.model.classifier.fit(reconstructed_x_train, y_train)

        return reconstructed_x_test, self.model


def train_autoencoder(x_train, nb_iter):
    """
    Trains an autoencoder model for input reconstruction.

    Args:
        x_train: Original training data.
        nb_iter: Number of iterations for training.

    Returns:
        Trained autoencoder model.
    """
    input_dim = x_train.shape[1]  # Assuming x_train is of shape (num_samples, num_features)
    encoding_dim = input_dim // 2  # Size of the encoded representation

    # Define the autoencoder model
    input_img = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu')(input_img)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer=Adam(), loss='binary_crossentropy')

    # Train the autoencoder
    autoencoder.fit(x_train, x_train, epochs=nb_iter, batch_size=256, shuffle=True)

    return autoencoder