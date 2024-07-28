from sklearn import tree
from Lab1.src.Classifiers import BaseClassifier


class DecisionTree(BaseClassifier.BaseClassifier):
    """
    Decision tree classifier implementation using DecisionTreeClassifier from sklearn.

    Attributes:
        DTBestParameters (dict): Best parameters found during optimization.
        metric_choose (str): Metric used for evaluation.
    """

    def __init__(self, x_train, y_train, x_test, y_test, metric_choose='*'):
        """
        Initializes the Decision Tree classifier.

        Args:
            metric_choose (str): Metric used for evaluation.
        """
        BaseClassifier.BaseClassifier.__init__(self, x_train, y_train, x_test, y_test, metric_choose)
        self.DTBestParameters = self.parameter_optimize()
        # self.DTBestParameters = self.get_optimized_paramater()
        self.k_fold_cross_validation(10, self.DTBestParameters, tree.DecisionTreeClassifier)

    def get_optimized_paramater(self):
        """
        Returns the best parameters found during optimization.

        Returns:
            dict: Best parameters found during optimization.
        """
        return {'accuracy': {'value': 0.936, 'parameters': {'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'entropy', 'max_depth': 20, 'max_features': None, 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'random_state': 0, 'splitter': 'best'}}}


    def parameter_optimize(self):
        """
        Performs parameter optimization for Decision Tree classifier.

        Returns:
            dict: Best parameters found during optimization.
        """
        best_parameters = {}

        for criterionArg in ["gini", "entropy"]:
            for sample_split in [2, 5, 10]:
                for min_samples_leaf in [1, 2, 5, 10]:
                    for maxDepthArg in [None, 5, 10, 15, 20, 25]:
                        metrics_y_test = []

                        parameters = {'criterion': criterionArg,
                                      'max_depth': maxDepthArg,
                                      'min_samples_split': sample_split,
                                      'min_samples_leaf': min_samples_leaf,
                                      'random_state': 0}

                        classifier = tree.DecisionTreeClassifier(**parameters)
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
        return self.DTBestParameters

    def __str__(self):
        """
        This special method returns a string representation of the object.
        It is called by the str() and print() functions.

        Returns:
            str: the string "Decision Tree" which represents the name
                 of the algorithm implemented by this class.
        """
        return "Decision Tree"