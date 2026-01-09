import matplotlib.pyplot as plt
import numpy as np
import os

# Output folder
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")

# Create synthetic values just for plots
epochs = np.arange(1, 11)
train_acc = np.linspace(0.50, 0.88, 10)
val_acc = np.linspace(0.48, 0.81, 10)

train_loss = np.linspace(1.2, 0.25, 10)
val_loss = np.linspace(1.3, 0.45, 10)

# ---- Accuracy Curve ----
plt.figure(figsize=(6,4))
plt.plot(epochs, train_acc, label="Train Accuracy")
plt.plot(epochs, val_acc, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.tight_layout()

acc_path = os.path.join(OUTPUT_DIR, "accuracy_curve.png")
plt.savefig(acc_path, dpi=200)
plt.close()

# ---- Loss Curve ----
plt.figure(figsize=(6,4))
plt.plot(epochs, train_loss, label="Train Loss")
plt.plot(epochs, val_loss, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.tight_layout()

loss_path = os.path.join(OUTPUT_DIR, "loss_curve.png")
plt.savefig(loss_path, dpi=200)
plt.close()

print("Saved:")
print(" -", acc_path)
print(" -", loss_path)
