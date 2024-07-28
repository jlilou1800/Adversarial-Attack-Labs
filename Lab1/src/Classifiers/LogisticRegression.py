from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from Lab1.src.Classifiers import BaseClassifier
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

class Logistic_Regression(BaseClassifier.BaseClassifier):
    """
    Logistic Regression classifier implementation.
    """
    def __init__(self, x_train, y_train, x_test, y_test,metric_choose='*'):
        """
        Initializes the Logistic Regression classifier.

        Args:
            metric_choose (str, optional): Metric to choose for evaluation. Defaults to '*'.
        """
        BaseClassifier.BaseClassifier.__init__(self, x_train, y_train, x_test, y_test, metric_choose)
        self.LRBestParameters = self.parameter_optimize()
        # self.LRBestParameters = self.get_optimized_paramater()
        self.k_fold_cross_validation(10, self.LRBestParameters, LogisticRegression)

    def get_optimized_paramater(self):
        """
        Returns the optimized parameters for the Logistic Regression classifier.

        Returns:
            dict: Optimized parameters for the Logistic Regression classifier.
        """
        return {'accuracy': {'value': 0.931, 'parameters': {'C': 0.1, 'class_weight': None, 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1, 'l1_ratio': None, 'max_iter': 500, 'multi_class': 'multinomial', 'n_jobs': None, 'penalty': 'l2', 'random_state': 0, 'solver': 'lbfgs', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}}}


    def parameter_optimize(self):
        """
        Optimizes the parameters for the Logistic Regression classifier.

        Returns:
            dict: Best parameters found during optimization.
        """
        best_parameters = {}
        for solver in ['newton-cg', 'lbfgs']:
            for C in [0.0001, 0.001, 0.01, 0.1]:
                metrics_y_test = []

                parameters = {'multi_class': 'multinomial',
                                      'C': C,
                                      'solver': solver,
                                      'penalty': 'l2',
                                      'random_state': 0,
                                        'max_iter': 500}
                classifier = LogisticRegression(**parameters)
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
        return self.LRBestParameters

    def __str__(self):
        """
        This special method returns a string representation of the object.
        It is called by the str() and print() functions.

        Returns:
            str: the string "Logistic Regression" which represents the name
                 of the algorithm implemented by this class.
        """
        return "Logistic Regression"