from sklearn.naive_bayes import GaussianNB
from Lab1.src.Classifiers import BaseClassifier


class BayesianNetwork(BaseClassifier.BaseClassifier):
    """
    Naive Bayesian network classifier implementation using GaussianNB.

    Attributes:
        BNBestParameters (dict): Best parameters found during optimization.
        metric_choose (str): Metric used for evaluation.
    """
    def __init__(self, x_train, y_train, x_test, y_test, metric_choose='*'):
        """
        Initializes the Bayesian Network classifier.

        Args:
            metric_choose (str): Metric used for evaluation.
        """
        BaseClassifier.BaseClassifier.__init__(self, x_train, y_train, x_test, y_test, metric_choose)
        self.BNBestParameters = self.parameter_optimize()
        self.k_fold_cross_validation(10, self.BNBestParameters, GaussianNB)

    # def get_optimized_paramater(self):
    #     return

    def parameter_optimize(self):
        best_parameters = {}

        metrics_y_test = []
        parameters = {}
        classifier = GaussianNB(**parameters)
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
        return self.BNBestParameters

    def __str__(self):
        """
        This special method returns a string representation of the object.
        It is called by the str() and print() functions.

        Returns:
            str: the string "Bayesian Network" which represents the name
                 of the algorithm implemented by this class.
        """
        return "Bayesian Network"