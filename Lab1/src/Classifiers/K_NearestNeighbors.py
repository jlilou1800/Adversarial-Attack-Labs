from sklearn.neighbors import KNeighborsClassifier
from Lab1.src.Classifiers import BaseClassifier


class K_NearestNeighbors(BaseClassifier.BaseClassifier):
    """
    K-Nearest Neighbors classifier implementation.
    """
    def __init__(self, x_train, y_train, x_test, y_test, metric_choose='*'):
        """
        Initializes the K-Nearest Neighbors classifier.

        Args:
            metric_choose (str, optional): Metric to choose for evaluation. Defaults to '*'.
        """
        BaseClassifier.BaseClassifier.__init__(self, x_train, y_train, x_test, y_test, metric_choose)
        self.KNNBestParameters = self.parameter_optimize()
        # self.KNNBestParameters = self.get_optimized_paramater()
        self.k_fold_cross_validation(10, self.KNNBestParameters, KNeighborsClassifier)

    def get_optimized_paramater(self):
        """
        Returns the optimized parameters for the K-Nearest Neighbors classifier.

        Returns:
            dict: Optimized parameters for the K-Nearest Neighbors classifier.
        """
        return {'accuracy': {'value': 0.938, 'parameters': {'algorithm': 'auto', 'leaf_size': 30, 'metric': 'minkowski', 'metric_params': None, 'n_jobs': None, 'n_neighbors': 18, 'p': 1, 'weights': 'distance'}}}


    def parameter_optimize(self):
        """
        Optimizes the parameters for the K-Nearest Neighbors classifier.

        Returns:
            dict: Best parameters found during optimization.
        """
        best_parameters = {}
        for n_neighbors in range(1, 20):
            for weights in ['uniform', 'distance']:
                for p in range(1, 2):
                    metrics_y_test = []

                    parameters = {'n_neighbors': n_neighbors,
                                      'weights': weights,
                                      'p': p}
                    classifier = KNeighborsClassifier(**parameters)
                    classifier.fit(self.x_train, self.y_train)
                    results_test = classifier.predict(self.x_test)
                    metrics_y_true = self.y_test
                    metrics_y_test = metrics_y_test + list(results_test)

                    evaluated_test_metrics = self.evaluationMetrics(metrics_y_true, metrics_y_test)

                    for key in evaluated_test_metrics:
                        if key not in best_parameters.keys():
                            best_parameters[key] = {"value": 0, "parameters": None}
                        if best_parameters[key]["value"] <= evaluated_test_metrics[key]:
                            best_parameters[key]["value"] = evaluated_test_metrics[key]
                            best_parameters[key]["parameters"] = classifier.get_params()
        return best_parameters

    def get_best_parameters(self):
        """
        Returns the best parameters found during optimization.

        Returns:
            dict: Best parameters found during optimization.
        """
        return self.KNNBestParameters

    def __str__(self):
        """
        This special method returns a string representation of the object.
        It is called by the str() and print() functions.

        Returns:
            str: the string "K Nearest Neighbors" which represents the name
                 of the algorithm implemented by this class.
        """
        return "K Nearest Neighbors"
