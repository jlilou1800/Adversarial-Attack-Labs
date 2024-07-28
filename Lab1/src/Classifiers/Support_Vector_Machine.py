from sklearn import svm
from Lab1.src.Classifiers import BaseClassifier


class Support_Vector_Machine(BaseClassifier.BaseClassifier):
    """
    Support Vector Machine classifier implementation.
    """
    def __init__(self, x_train, y_train, x_test, y_test, metric_choose='*'):
        """
        Initializes the Support Vector Machine classifier.

        Args:
            metric_choose (str, optional): Metric to choose for evaluation. Defaults to '*'.
        """
        BaseClassifier.BaseClassifier.__init__(self, x_train, y_train, x_test, y_test, metric_choose)
        self.SVCBestParameters = self.parameter_optimize()
        # self.SVCBestParameters = self.get_optimized_paramater()
        self.k_fold_cross_validation(10, self.SVCBestParameters, svm.SVC)

    def get_optimized_paramater(self):
        """
        Returns the optimized parameters for the Support Vector Machine classifier.

        Returns:
            dict: Optimized parameters for the Support Vector Machine classifier.
        """
        return {'accuracy': {'value': 0.965, 'parameters': {'C': 100, 'break_ties': False, 'cache_size': 200, 'class_weight': None, 'coef0': 0.0, 'decision_function_shape': 'ovr', 'degree': 3, 'gamma': 'scale', 'kernel': 'rbf', 'max_iter': -1, 'probability': True, 'random_state': 0, 'shrinking': True, 'tol': 0.001, 'verbose': False}}}


    def parameter_optimize(self):
        """
        Optimizes the parameters for the Support Vector Machine classifier.

        Returns:
            dict: Best parameters found during optimization.
        """
        best_parameters = {}
        for C in [0.1, 1, 10, 100]:
            for gamma in ['auto', 'scale']:
                metrics_y_test = []
                parameters = {'kernel': 'rbf',
                              'gamma': gamma,
                              'C': C,
                              'probability': True,
                              'random_state': 0}
                classifier = svm.SVC(**parameters)
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
        return self.SVCBestParameters

    def __str__(self):
        """
        This special method returns a string representation of the object.
        It is called by the str() and print() functions.

        Returns:
            str: the string "Support Vector Machine" which represents the name
                 of the algorithm implemented by this class.
        """
        return "Support Vector Machine"