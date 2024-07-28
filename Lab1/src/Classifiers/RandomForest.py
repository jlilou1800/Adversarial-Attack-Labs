from sklearn.ensemble import RandomForestClassifier
from Lab1.src.Classifiers import BaseClassifier


class RandomForest(BaseClassifier.BaseClassifier):
	"""
	Random Forest classifier implementation.
	"""
	def __init__(self, x_train, y_train, x_test, y_test, metric_choose='*'):
		"""
		Initializes the Random Forest classifier.

		Args:
		    metric_choose (str, optional): Metric to choose for evaluation. Defaults to '*'.
		"""
		BaseClassifier.BaseClassifier.__init__(self, x_train, y_train, x_test, y_test, metric_choose)
		RFBestParameters = self.parameter_optimize()
		# RFBestParameters = self.get_optimized_paramater()
		self.k_fold_cross_validation(10, RFBestParameters, RandomForestClassifier)

	def get_optimized_paramater(self):
		"""
		Returns the optimized parameters for the Random Forest classifier.

		Returns:
		    dict: Optimized parameters for the Random Forest classifier.
		"""
		return {'accuracy': {'value': 0.911, 'parameters': {'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'gini', 'max_depth': 14, 'max_features': 'sqrt', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 45, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 70, 'n_jobs': None, 'oob_score': False, 'random_state': 0, 'verbose': 0, 'warm_start': False}}}


	def parameter_optimize(self):
		"""
		Optimizes the parameters for the Random Forest classifier.

		Returns:
		    dict: Best parameters found during optimization.
		"""
		best_parameters = {}

		for criterionArg in ["gini", "entropy"]:
			for n_estimators in range(50, 80, 10):
				for sample_split in range(45, 55, 5):
					for maxDepthArg in range(9, 15):
						metrics_y_test = []

						parameters = {'criterion': criterionArg,
									  'max_depth': maxDepthArg,
									  'n_estimators': n_estimators,
									  'min_samples_split': sample_split,
									  'random_state': 0}

						classifier = RandomForestClassifier(**parameters)
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
		    str: the string "Random Forest" which represents the name
		         of the algorithm implemented by this class.
		"""
		return "Random Forest"