from sklearn.ensemble import AdaBoostClassifier
from Lab1.src.Classifiers import BaseClassifier
from numpy import arange


class AdaBoost(BaseClassifier.BaseClassifier):
	"""
	AdaBoost classifier implementation using sklearn's AdaBoostClassifier.

	Attributes:
	    metric_choose (str): Metric used for evaluation.
	"""
	def __init__(self, x_train, y_train, x_test, y_test, metric_choose='*'):
		"""
		Initializes the AdaBoost classifier.

		Args:
		    metric_choose (str): Metric used for evaluation.
		"""
		BaseClassifier.BaseClassifier.__init__(self, x_train, y_train, x_test, y_test, metric_choose)
		ABBestParameters = self.parameter_optimize()
		self.k_fold_cross_validation(10, ABBestParameters, AdaBoostClassifier)

	def get_optimized_paramater(self):
		"""
		Returns the optimized parameters for AdaBoost.

		Returns:
		    dict: Optimized parameters for AdaBoost.
		"""
		return {'n_estimators': 31, 'algorithm': 'SAMME.R', 'learning_rate': 0.5, 'random_state': 0}

	def parameter_optimize(self):
		"""
		Performs parameter optimization for AdaBoost.

		Returns:
		    dict: Best parameters found during optimization.
		"""
		best_parameters = {}
		for n_estimators_value in range(1, 52, 10):
			for algorithm_value in ["SAMME", "SAMME.R"]:
				for learning_rate_value in arange(0.5, 2.5, 0.5):
					metrics_y_test = []
					parameters = {'n_estimators': n_estimators_value,
								  'algorithm': algorithm_value,
								  'learning_rate': learning_rate_value,
								  'random_state': 0}
					classifier = AdaBoostClassifier(**parameters)
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

	def __str__(self):
		"""
		This special method returns a string representation of the object.
		It is called by the str() and print() functions.

		Returns:
			str: the string "Adaptive Boosting" which represents the name
				 of the algorithm implemented by this class.
		"""
		return "Adaptive Boosting"
