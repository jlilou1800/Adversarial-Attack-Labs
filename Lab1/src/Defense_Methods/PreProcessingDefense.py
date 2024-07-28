import copy

from sklearn.decomposition import PCA

class PreProcessingDefense:
    """
    Defense method that preprocesses input data to enhance model robustness.
    """

    def __init__(self, model):
        """
        Initializes the PreProcessingDefense defense.
        """
        self.pca = None
        self.model = copy.deepcopy(model)

    def defend(self, x_train, y_train, x_test):
        """
        Enhances model robustness through preprocessing defense.

        Args:
            x_train: Original training data.

        Returns:
            Transformed input data.
        """
        self.pca = PCA(n_components=0.95)  # Keep 95% of variance
        x_train_transformed = self.pca.fit_transform(x_train)
        self.model.classifier.fit(x_train_transformed, y_train)
        x_test_transformed = self.transform(x_test)

        return x_test_transformed, self.model

    def transform(self, x):
        """
        Transforms the input data using the trained PCA.

        Args:
            x: Input data to be transformed.

        Returns:
            Transformed input data.
        """
        return self.pca.transform(x) if self.pca else x