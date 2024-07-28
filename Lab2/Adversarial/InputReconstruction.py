import numpy as np


class InputReconstruction:
    """
    A class to handle input reconstruction using an autoencoder to detect adversarial examples.

    Attributes:
    autoencoder : object
        The autoencoder model used for reconstructing inputs.
    threshold : float
        The threshold value for determining if an example is adversarial based on reconstruction error.
    """

    def __init__(self, autoencoder, threshold):
        """
        Initializes the InputReconstruction class with the provided autoencoder model and threshold.

        Parameters:
        autoencoder : object
            The autoencoder model used to reconstruct inputs.
        threshold : float
            The threshold value for reconstruction error to detect adversarial examples.
        """
        self.autoencoder = autoencoder
        self.threshold = threshold

    def input_reconstruction(self, x_test):
        """
        Reconstructs the input examples using the autoencoder.

        Parameters:
        x_test : array-like
            The test input data to be reconstructed.

        Returns:
        recon_x : array-like
            The reconstructed input data.
        """
        recon_x = self.autoencoder.predict(x_test)
        return recon_x

    def calculate_reconstruction_error(self, x_test):
        """
        Calculates the reconstruction error between the original input and the reconstructed input.

        Parameters:
        x_test : array-like
            The original test input data.

        Returns:
        reconstruction_error : array-like
            The calculated reconstruction error for each input example.
        """
        recon_x = self.input_reconstruction(x_test)
        reconstruction_error = np.linalg.norm(x_test - recon_x, ord=2, axis=1)
        return reconstruction_error

    def detect_adversarial_examples(self, x_test):
        """
        Detects adversarial examples based on the reconstruction error and the predefined threshold.

        Parameters:
        x_test : array-like
            The test input data to be evaluated.

        Returns:
        is_adversarial : array-like
            A boolean array indicating whether each input example is adversarial (True) or not (False).
        """
        reconstruction_error = self.calculate_reconstruction_error(x_test)
        is_adversarial = reconstruction_error > self.threshold
        return is_adversarial
