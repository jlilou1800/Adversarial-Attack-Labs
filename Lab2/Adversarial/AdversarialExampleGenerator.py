import tensorflow as tf
from cleverhans.tf2.attacks.basic_iterative_method import basic_iterative_method
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.tf2.attacks.momentum_iterative_method import momentum_iterative_method
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from tensorflow.keras import layers, models
from Lab2.Adversarial.DefenseGAN import DefenseGAN
from Lab2.Adversarial.InputReconstruction import InputReconstruction


class AdversarialExampleGenerator:
    """
    A class for generating adversarial examples using various attack methods and defenses.
    """

    def __init__(self, org_model, input_size):
        """
        Initializes the AdversarialExampleGenerator with the original model and input size.

        Args:
            org_model (tf.keras.Model): The original model.
            input_size (int): The size of the input features.
        """
        self.org_model = org_model
        self.logits_model = tf.keras.Model(org_model.input, org_model.layers[-1].output)
        self.input_size = input_size

    def generate_fgsm_attack(self, x_test, epsilon):
        """
        Generates adversarial examples using the Fast Gradient Sign Method (FGSM).

        Args:
            x_test (numpy.ndarray): Input test data.
            epsilon (float): Perturbation size for FGSM.

        Returns:
            numpy.ndarray: Adversarial examples generated using FGSM.
        """
        adv_fgsm_x = fast_gradient_method(self.logits_model, x_test, epsilon, np.inf)
        return adv_fgsm_x

    def generate_bim_attack(self, x_test, epsilon):
        """
        Generates adversarial examples using the Basic Iterative Method (BIM).

        Args:
            x_test (numpy.ndarray): Input test data.
            epsilon (float): Perturbation size for BIM.

        Returns:
            numpy.ndarray: Adversarial examples generated using BIM.
        """
        adv_bim_x = basic_iterative_method(self.logits_model, x_test, epsilon, 0.01, 20, np.inf)
        return adv_bim_x

    def generate_pgd_attack(self, x_test, epsilon):
        """
        Generates adversarial examples using the Projected Gradient Descent Method (PGD).

        Args:
            x_test (numpy.ndarray): Input test data.
            epsilon (float): Perturbation size for PGD.

        Returns:
            numpy.ndarray: Adversarial examples generated using PGD.
        """
        adv_pgd_x = projected_gradient_descent(self.logits_model, x_test, epsilon, 0.01, 40, np.inf)
        return adv_pgd_x

    def generate_mim_attack(self, x_test, epsilon):
        """
        Generates adversarial examples using the Momentum Iterative Method (MIM).

        Args:
            x_test (numpy.ndarray): Input test data.
            epsilon (float): Perturbation size for MIM.

        Returns:
            numpy.ndarray: Adversarial examples generated using MIM.
        """
        adv_mim_x = momentum_iterative_method(self.logits_model, x_test, epsilon, 0.1, 10, np.inf)
        return adv_mim_x

    def defense_gan(self, x_test):
        """
        Implements the Defense-GAN defense method.

        Args:
            x_test (numpy.ndarray): Input test data.

        Returns:
            numpy.ndarray: Defended examples using Defense-GAN.
        """
        defense_gan = DefenseGAN(self.input_size)
        defended_x = defense_gan.defense_gan(x_test)
        return defended_x

    def input_reconstruction(self, x_test):
        """
        Implements the Input Reconstruction (Manifold Analysis) defense method.

        Args:
            x_test (numpy.ndarray): Input test data.

        Returns:
            numpy.ndarray: Defended examples using Input Reconstruction.
        """
        autoencoder = build_autoencoder(self.input_size)
        autoencoder.fit(x_test, x_test, epochs=10, batch_size=256, shuffle=True, verbose=0)
        input_reconstruction = InputReconstruction(autoencoder, threshold=0.1)
        defended_x = input_reconstruction.input_reconstruction(x_test)
        return defended_x


def build_autoencoder(input_dim):
    """
    Builds and compiles an autoencoder model for input data of specified dimensionality.

    Parameters:
    input_dim : int
        The dimensionality of the input data.

    Returns:
    autoencoder : keras.models.Model
        The compiled autoencoder model.
    """
    # Define the input layer with the specified input dimension
    input_layer = layers.Input(shape=(input_dim,))

    # Define the encoding layer that reduces the input dimension by half
    encoded = layers.Dense(input_dim // 2, activation='relu')(input_layer)

    # Define the decoding layer that reconstructs the input dimension from the encoded representation
    decoded = layers.Dense(input_dim, activation='sigmoid')(encoded)

    # Construct the autoencoder model with the input layer and the decoded output
    autoencoder = models.Model(input_layer, decoded)

    # Compile the autoencoder model using the Adam optimizer and binary crossentropy loss function
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return autoencoder
