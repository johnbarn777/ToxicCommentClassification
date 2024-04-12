import matplotlib
matplotlib.use('Agg')  # This should be called before importing pyplot

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np

def plot_roc_curve(true_labels, probabilities, label_names):
    roc_aucs = []

    # Set up the plot
    plt.figure(figsize=(10, 8))
    colors = iter(plt.cm.rainbow(np.linspace(0, 1, len(label_names))))
    print(true_labels)
    print(probabilities)
    # Compute and plot the ROC curve for each label
    for i, label in enumerate(label_names):
        fpr, tpr, _ = roc_curve(true_labels[:, i], probabilities[:, i])
        roc_auc = auc(fpr, tpr)
        roc_aucs.append(roc_auc)
        color = next(colors)
        plt.plot(fpr, tpr, color=color, lw=2, label=f'ROC curve of class {label} (area = {roc_auc:.2f})')
        plt.show()

    # Plot the line of no skill
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    # Customize the plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Each Label')
    plt.legend(loc="lower right")

    # Show plot
    plt.show()
    return roc_aucs

# You need to have true_labels and probabilities ready before calling this function.
# true_labels should be a binary matrix with shape (num_samples, num_labels)
# probabilities should be a matrix of probabilities with shape (num_samples, num_labels)
# label_names should be a list of strings corresponding to each label

# roc_aucs = plot_roc_curve(true_labels, probabilities, label_names)

