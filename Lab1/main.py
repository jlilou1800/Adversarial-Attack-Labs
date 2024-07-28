# coding: utf8
import concurrent.futures
import os
import sys
import torch
from scipy.optimize._linesearch import LineSearchWarning
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import numpy as np
import warnings

from Lab1.src.Attack_Methods import (
    ProjectedGradientDescentMethod,
    BasicIterativeMethod,
    OneStepTargetClassMethod,
    IterativeLeastLikelyClassMethod,
    CarliniWagnerMethod
)
from Lab1.src.Classifiers import (
    RandomForest,
    AdaptiveBoosting,
    K_NearestNeighbors,
    LogisticRegression,
    Support_Vector_Machine,
    BayesianNetwork
)
from Lab1.src.Defense_Methods import (
    AdversarialTraining,
    DefensiveDistillation,
    InputReconstruction,
    ModelRobustifying
)
from src import Database
from src.Attack_Methods import FastGradientSignMethod
from src.Classifiers import DecisionTree
from src.Defense_Methods import PreProcessingDefense

# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Suppress specific warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UndefinedMetricWarning)
warnings.simplefilter(action='ignore', category=LineSearchWarning)

# Constants for image dimensions
COLUMN = 30
LINE = 30

# Global dictionary to store results
results_dict = {}

def main():
    """
    Main function to perform database creation, model training, and adversarial attack and defense evaluation.
    """

    # epsilons = [.5]
    epsilons = [.1, .5, 1.5, 5, 15]
    attack_methods = [
        FastGradientSignMethod.FastGradientSignMethod(LINE, COLUMN),
        ProjectedGradientDescentMethod.ProjectedGradientDescentMethod(LINE, COLUMN),
        BasicIterativeMethod.BasicIterativeMethod(LINE, COLUMN),
        OneStepTargetClassMethod.OneStepTargetClassMethod(LINE, COLUMN),
        IterativeLeastLikelyClassMethod.IterativeLeastLikelyClassMethod(LINE, COLUMN),
        CarliniWagnerMethod.CarliniWagnerMethod(LINE, COLUMN)
    ]

    print("<" * 25, "Database creation", ">" * (60 - len("Database creation")))

    # Uncomment the following lines if database creation and k-fold cross-validation are needed
    # db = Database.Database(2500)
    # db.createDb()
    # fold = 5
    # db.kFoldCrossValidation(fold, 1)
    # db.define_labels()

    x_train, y_train, x_test, y_test = loadData()

    print("Database created !", "-" * (70 - len("Database created !")))

    print("<" * 25, "ML creation and Training", ">" * (60 - len("ML creation and Training")))

    models = parameter_testing(x_train, y_train, x_test, y_test)
    print("Classifiers created and trained !", "-" * (70 - len("Classifiers created and trained !")))

    print("<" * 25, "Adversarial Attack Creation", ">" * (60 - len("Adversarial Attack Creation")))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for model in models:
            for attack_method in attack_methods:
                for eps in epsilons:
                    futures.append(executor.submit(process_model, model, attack_method, eps, x_train, y_train, x_test, y_test))

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"Generated an exception: {exc}")

    # Print the results dictionary
    with open("src/Results/res.txt", "w") as output_file:
        sys.stdout = output_file
        print(results_dict)

def process_model(model, attack_method, eps, x_train, y_train, x_test, y_test):
    """
    Process a single model with a specific attack method and epsilon value.

    Args:
        model: The model to process.
        attack_method: The attack method to use.
        eps: The epsilon value for the attack.
        x_train (array-like): Training data features.
        y_train (array-like): Training data labels.
        x_test (array-like): Testing data features.
        y_test (array-like): Testing data labels.
    """
    defense_methods = [
        AdversarialTraining.AdversarialTraining(model),
        DefensiveDistillation.DefensiveDistillation(model),
        InputReconstruction.InputReconstruction(model),
        ModelRobustifying.ModelRobustifying(model),
        PreProcessingDefense.PreProcessingDefense(model)
    ]

    x_adv = attack_method.generate_adversarial_set(x_test, y_test, eps)
    y_pred = model.predict(x_adv, y_test)[1]
    attack_metrics = evaluation_metrics(y_test, y_pred)
    results_dict[f"{model}_Attack_{attack_method}_eps_{eps}"] = attack_metrics

    with concurrent.futures.ThreadPoolExecutor() as executor:
        defense_futures = []
        for defense_method in defense_methods:
            defense_futures.append(executor.submit(process_defense, model, defense_method, x_train, y_train, x_adv, x_test, y_test, eps, attack_method))

        for defense_future in concurrent.futures.as_completed(defense_futures):
            try:
                defense_future.result()
            except Exception as exc:
                print(f"Generated an exception: {exc}")

def process_defense(model, defense_method, x_train, y_train, x_adv, x_test, y_test, eps, attack_method):
    """
    Process a single defense method for a given model and adversarial examples.

    Args:
        model: The model to process.
        defense_method: The defense method to apply.
        x_train (array-like): Training data features.
        y_train (array-like): Training data labels.
        x_adv (array-like): Adversarial examples.
        x_test (array-like): Testing data features.
        y_test (array-like): Testing data labels.
        eps: The epsilon value for the attack.
        attack_method: The attack method used to generate adversarial examples.
    """
    print("*" * 10, defense_method, "*" * 10)
    if isinstance(defense_method, AdversarialTraining.AdversarialTraining):
        parameter = (x_train, y_train, eps, attack_method)
    elif isinstance(defense_method, DefensiveDistillation.DefensiveDistillation):
        parameter = (x_train, y_train, 10)
    elif isinstance(defense_method, InputReconstruction.InputReconstruction):
        parameter = (x_train, y_train, x_adv, 3)
    elif isinstance(defense_method, ModelRobustifying.ModelRobustifying):
        parameter = (x_train, y_train, eps, 5)
    elif isinstance(defense_method, PreProcessingDefense.PreProcessingDefense):
        parameter = (x_train, y_train, x_test)
    else:
        raise ValueError("Unknown defense method: {}".format(defense_method))

    if isinstance(defense_method, InputReconstruction.InputReconstruction) or \
            isinstance(defense_method, PreProcessingDefense.PreProcessingDefense):
        modified_x_adv, training_model = defense_method.defend(*parameter)
        y_pred = training_model.predict(modified_x_adv, y_test)[1]
    else:
        training_model = defense_method.defend(*parameter)
        y_pred = training_model.predict(x_adv, y_test)[1]

    defense_metrics = evaluation_metrics(y_test, y_pred)
    results_dict[f"{model}_Defense_{defense_method}_eps_{eps}"] = defense_metrics

def evaluation_metrics(y_test, y_pred):
    """
    Evaluate and return various performance metrics.

    Args:
        y_test (array-like): True labels.
        y_pred (array-like): Predicted labels.

    Returns:
        dict: Dictionary containing the evaluation metrics.
    """
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='weighted'),
        "Recall": recall_score(y_test, y_pred, average='weighted'),
        "F-measure": f1_score(y_test, y_pred, average='weighted')
    }
    # print(metrics)
    return metrics

def parameter_testing(x_train, y_train, x_test, y_test):
    """
    Create and return classifiers with optimized parameters.

    Args:
        x_train (array-like): Training data features.
        y_train (array-like): Training data labels.
        x_test (array-like): Testing data features.
        y_test (array-like): Testing data labels.

    Returns:
        list: List of trained classifiers.
    """
    def create_classifier(classifier_class, name):
        classifier = classifier_class(x_train, y_train, x_test, y_test, 'f_measure')
        return classifier

    classifiers = [
        (DecisionTree.DecisionTree, "Decision Tree"),
        (RandomForest.RandomForest, "Random Forest"),
        (AdaptiveBoosting.AdaBoost, "AdaBoost"),
        (K_NearestNeighbors.K_NearestNeighbors, "K Nearest Neighbors"),
        (LogisticRegression.Logistic_Regression, "Logistic Regression"),
        (Support_Vector_Machine.Support_Vector_Machine, "Support Vector Machine"),
        (BayesianNetwork.BayesianNetwork, "Bayesian Network")
    ]

    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_classifier = {executor.submit(create_classifier, cls, name): name for cls, name in classifiers}
        for future in concurrent.futures.as_completed(future_to_classifier):
            classifier_name = future_to_classifier[future]
            try:
                result = future.result()
                results.append(result)
                print(f"{classifier_name} - Parameter setting completed.")
            except Exception as exc:
                print(f"{classifier_name} generated an exception: {exc}")

    return results

def clearRepository(repo_name):
    """
    Clear the specified repository by deleting all files.

    Args:
        repo_name (str): Name of the repository to clear.
    """
    try:
        for file in os.listdir(repo_name):
            file_path = os.path.join(repo_name, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)
    except Exception as e:
        print(e)

def loadData():
    """
    Load training and testing data from files.

    Returns:
        tuple: Loaded training and testing data (x_train, y_train, x_test, y_test).
    """
    x_train = np.loadtxt("src/dataset_flattened/X_train.txt")
    y_train = np.loadtxt("src/dataset_flattened/Y_train.txt")
    x_test = np.loadtxt("src/dataset_flattened/X_test.txt")
    y_test = np.loadtxt("src/dataset_flattened/Y_test.txt")

    return x_train, y_train, x_test, y_test

if __name__ == "__main__":
    main()
