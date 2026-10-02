import pandas as pd
import matplotlib.pyplot as plt
import os

metrics_file = "outputs/clean_metrics.csv"
output_file = "outputs/training_curves.png"

df = pd.read_csv(metrics_file)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Loss
axes[0].plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o')
axes[0].plot(df['epoch'], df['val_loss'], label='Val Loss', marker='o')
axes[0].set_title('Loss over Epochs')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True)

# Plot 2: AUC and Accuracy
axes[1].plot(df['epoch'], df['macro_auc'], label='Val Macro AUC', marker='o')
axes[1].plot(df['epoch'], df['overall_accuracy'], label='Val Accuracy', marker='o')
if 'train_accuracy' in df.columns:
    axes[1].plot(df['epoch'], df['train_accuracy'], label='Train Accuracy', marker='o')
axes[1].set_title('AUC & Accuracy over Epochs')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Score')
axes[1].legend()
axes[1].grid(True)

# Plot 3: F1, Precision, Recall
axes[2].plot(df['epoch'], df['macro_f1'], label='Macro F1', marker='o')
axes[2].plot(df['epoch'], df['macro_precision'], label='Macro Precision', marker='o')
axes[2].plot(df['epoch'], df['macro_recall'], label='Macro Recall', marker='o')
axes[2].set_title('F1, Precision & Recall over Epochs')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Score')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.savefig(output_file, dpi=150)
print(f"Saved curves to {output_file}")
