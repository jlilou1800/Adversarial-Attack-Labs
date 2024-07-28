import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load the data
data_path = '../Results/res lab2.txt'
data = pd.read_csv(data_path, sep=r'\s{2,}', engine='python')

# Strip whitespace from column names
data.columns = data.columns.str.strip()

# Check the column names to ensure they are correct
print(data.columns)

# Define a function to plot the metrics
def plot_metrics(data, metric):
    plt.figure(figsize=(14, 8))
    sns.lineplot(data=data, x="Epsilon", y=metric, hue="Defense", style="Attack", markers=True, dashes=False)
    plt.title(f'{metric} vs Epsilon for Different Defenses and Attacks', fontsize=16)
    plt.xlabel('Epsilon', fontsize=14)
    plt.ylabel(metric, fontsize=14)
    plt.legend(title='Defense', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{metric}_vs_Epsilon.png")
    plt.show()

# Plot the metrics
metrics = ["Accuracy", "Recall", "Precision", "F1 Score"]
for metric in metrics:
    plot_metrics(data, metric)
