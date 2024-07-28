import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data for visualization
data = {
    "Method": ["Original", "FGSM_0.1", "FGSM_0.1_DefenseGAN", "FGSM_0.1_InputReconstruction",
               "FGSM_0.1_AdversarialTraining", "FGSM_0.1_ModelRobustifying",
               "FGSM_0.5", "FGSM_0.5_DefenseGAN", "FGSM_0.5_InputReconstruction",
               "FGSM_0.5_AdversarialTraining", "FGSM_0.5_ModelRobustifying"],
    "Accuracy": [0.9956, 0.9813, 0.5581, 0.4556, 0.9857, 0.9916,
                 0.2972, 0.5804, 0.5581, 0.9916, 0.9916]
}

df = pd.DataFrame(data)

# Plotting the bar chart for Accuracy
plt.figure(figsize=(14, 7))
plt.bar(df["Method"], df["Accuracy"], color='skyblue')
plt.xlabel('Method', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.title('Accuracy of Different Methods', fontsize=16)
plt.xticks(rotation=45, ha="right", fontsize=12)
plt.tight_layout()  # Adjust layout to make sure everything fits without overlap
plt.show()

# Data for visualization
data = {
    "Method": ["Original", "FGSM_0.1", "FGSM_0.1_DefenseGAN", "FGSM_0.1_InputReconstruction",
               "FGSM_0.1_AdversarialTraining", "FGSM_0.1_ModelRobustifying",
               "FGSM_0.5", "FGSM_0.5_DefenseGAN", "FGSM_0.5_InputReconstruction",
               "FGSM_0.5_AdversarialTraining", "FGSM_0.5_ModelRobustifying"],
    "F1_Score": [0.9956, 0.9815, 0.5104, 0.4260, 0.9850, 0.9914,
                 0.1923, 0.5632, 0.5104, 0.9914, 0.9914]
}

df = pd.DataFrame(data)

# Plotting the bar chart for F1 Score
plt.figure(figsize=(14, 7))
plt.bar(df["Method"], df["F1_Score"], color='lightgreen')
plt.xlabel('Method', fontsize=14)
plt.ylabel('F1 Score', fontsize=14)
plt.title('F1 Score of Different Methods', fontsize=16)
plt.xticks(rotation=45, ha="right", fontsize=12)
plt.tight_layout()  # Adjust layout to make sure everything fits without overlap
plt.show()

# Example confusion matrix data
conf_matrix = np.array([[10874, 15, 0, 0],
                        [18, 17526, 7, 17],
                        [6, 25, 402, 8],
                        [1, 26, 4, 178]])

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGnBu")
plt.title('Confusion Matrix: Original Model', fontsize=16)
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)
plt.tight_layout()  # Adjust layout to make sure everything fits without overlap
plt.show()
