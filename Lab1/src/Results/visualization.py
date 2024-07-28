import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re


def load_results_from_txt(file_path):
    """
    Load results from a text file into a pandas DataFrame and filter by epsilon.

    Args:
        file_path (str): The path to the text file containing the results.

    Returns:
        pd.DataFrame: A DataFrame containing the filtered results.
    """
    with open(file_path, 'r') as file:
        data = eval(file.read())  # Safely evaluate the string to a dictionary

    records = []
    for key, metrics in data.items():
        match = re.match(r"(.+?)_(Attack|Defense)_<(.+?)>_eps_(.+)", key)
        if match:
            classifier_name, attack_or_defense, method, epsilon = match.groups()
            if float(epsilon) == 0.5:  # Filter by epsilon 0.5
                record = {
                    'Classifier': classifier_name,
                    'Attack/Defense': attack_or_defense,
                    'Method': method.split('.')[-1].split()[0],  # Get only the method class name
                    'Epsilon': float(epsilon),
                    'Accuracy': metrics['Accuracy'],
                    'Precision': metrics['Precision'],
                    'Recall': metrics['Recall'],
                    'F1 Score': metrics['F-measure']
                }
                records.append(record)

    df = pd.DataFrame(records)
    return df


def plot_results(df, classifier_name):
    """
    Plot the results for a given classifier.

    Args:
        df (pd.DataFrame): The DataFrame containing the results.
        classifier_name (str): The name of the classifier to plot.
    """
    classifier_df = df[df['Classifier'] == classifier_name]

    plt.figure(figsize=(14, 7))
    sns.barplot(data=classifier_df, x='Method', y='Accuracy', hue='Attack/Defense')
    plt.title(f'Accuracy of {classifier_name} under different attacks and defenses (Epsilon = 0.5)')
    plt.xlabel('Method')
    plt.ylabel('Accuracy')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 7))
    sns.barplot(data=classifier_df, x='Method', y='F1 Score', hue='Attack/Defense')
    plt.title(f'F1 Score of {classifier_name} under different attacks and defenses (Epsilon = 0.5)')
    plt.xlabel('Method')
    plt.ylabel('F1 Score')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def main():
    """
    Main function to create all graphs.
    """
    results_df = load_results_from_txt("all.txt")
    classifiers = ["Decision Tree", "K Nearest Neighbors", "Random Forest", "Support Vector Machine"]
    for clr in classifiers:
        plot_results(results_df, clr)


if __name__ == "__main__":
    main()
