
import os
import time
import numpy as np
import torch
import torch.nn as nn

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data_augmentation import (
    train_loader,
    val_loader,
    CLASS_NAMES
)

from model import CBAMResNet18


# ==========================================
# 1. CONFIGURATION
# ==========================================

NUM_CLASSES = 10

NUM_EPOCHS = 20  # Initial GPU test

LEARNING_RATE = 0.0001

WEIGHT_DECAY = 0.0001

PATIENCE = 5

MODEL_SAVE_PATH = "models/best_cbam_resnet18.pth"

HISTORY_SAVE_PATH = "results/training_history.npz"


# ==========================================
# 2. GPU CONFIGURATION
# ==========================================

if not torch.cuda.is_available():

    raise RuntimeError(
        "CUDA is not available! "
        "Check your PyTorch CUDA installation."
    )

DEVICE = torch.device("cuda:0")

torch.cuda.set_device(0)

torch.cuda.empty_cache()


# ==========================================
# 3. CREATE OUTPUT DIRECTORIES
# ==========================================

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


# ==========================================
# 4. GPU VERIFICATION
# ==========================================

print("\nCBAM-RESNET18 MODEL TRAINING")
print("----------------------------")

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Device: {DEVICE}")

print(f"GPU Name: {torch.cuda.get_device_name(0)}")

gpu_memory = (
    torch.cuda.get_device_properties(0).total_memory
    / (1024 ** 3)
)

print(f"GPU Memory: {gpu_memory:.2f} GB")


# ==========================================
# 5. INITIALIZE MODEL
# ==========================================

model = CBAMResNet18(
    num_classes=NUM_CLASSES,
    pretrained=True
)

model = model.to(DEVICE)

print("\nMODEL DEVICE CHECK")

print(
    "Model parameters are on:",
    next(model.parameters()).device
)


# ==========================================
# 6. LOAD CLASS WEIGHTS
# ==========================================

class_weights = np.load(
    "dataset/galaxy10_class_weights.npy"
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
).to(DEVICE)

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ==========================================
# 7. OPTIMIZER AND SCHEDULER
# ==========================================

optimizer = Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)


# ==========================================
# 8. TRAINING FUNCTION
# ==========================================

def train_one_epoch(model, loader):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        # Transfer data to NVIDIA GPU
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = torch.argmax(outputs, dim=1)

        correct += (predictions == labels).sum().item()

        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ==========================================
# 9. VALIDATION FUNCTION
# ==========================================

def validate_model(model, loader):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            # Transfer data to NVIDIA GPU
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            predictions = torch.argmax(outputs, dim=1)

            correct += (predictions == labels).sum().item()

            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ==========================================
# 10. MODEL TRAINING
# ==========================================

best_val_loss = float("inf")

epochs_without_improvement = 0

history = {
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": []
}

start_time = time.time()

for epoch in range(NUM_EPOCHS):

    epoch_start = time.time()

    print(f"\nEPOCH {epoch + 1}/{NUM_EPOCHS}")
    print("----------------------------")

    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader
    )

    val_loss, val_accuracy = validate_model(
        model,
        val_loader
    )

    scheduler.step(val_loss)

    # Synchronize GPU before measuring elapsed time
    torch.cuda.synchronize()

    epoch_time = time.time() - epoch_start

    current_lr = optimizer.param_groups[0]["lr"]

    history["train_loss"].append(train_loss)
    history["train_accuracy"].append(train_accuracy)

    history["val_loss"].append(val_loss)
    history["val_accuracy"].append(val_accuracy)

    print(f"Training Loss: {train_loss:.4f}")
    print(f"Training Accuracy: {train_accuracy:.4f}")

    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    print(f"Learning Rate: {current_lr:.6f}")
    print(f"Epoch Time: {epoch_time:.2f} seconds")

    # GPU memory monitoring
    print(
        "GPU Memory Allocated: "
        f"{torch.cuda.memory_allocated(0) / (1024 ** 3):.2f} GB"
    )

    print(
        "GPU Memory Reserved: "
        f"{torch.cuda.memory_reserved(0) / (1024 ** 3):.2f} GB"
    )

    # Save best model based on validation loss
    if val_loss < best_val_loss:

        best_val_loss = val_loss

        epochs_without_improvement = 0

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "class_names": CLASS_NAMES
            },
            MODEL_SAVE_PATH
        )

        print("Best model checkpoint saved!")

    else:

        epochs_without_improvement += 1

        print(
            f"No improvement: "
            f"{epochs_without_improvement}/{PATIENCE}"
        )

    # Early stopping
    if epochs_without_improvement >= PATIENCE:

        print("\nEarly stopping triggered.")

        break


# ==========================================
# 11. SAVE TRAINING HISTORY
# ==========================================

np.savez(
    HISTORY_SAVE_PATH,
    train_loss=history["train_loss"],
    train_accuracy=history["train_accuracy"],
    val_loss=history["val_loss"],
    val_accuracy=history["val_accuracy"]
)


# ==========================================
# 12. TRAINING SUMMARY
# ==========================================

torch.cuda.synchronize()

total_time = time.time() - start_time

print("\nTRAINING COMPLETED")
print("----------------------------")

print(f"Best Validation Loss: {best_val_loss:.4f}")

print(
    f"Best Validation Accuracy: "
    f"{max(history['val_accuracy']):.4f}"
)

print(f"Total Training Time: {total_time:.2f} seconds")

print(f"Best Model Saved: {MODEL_SAVE_PATH}")

print(f"Training History Saved: {HISTORY_SAVE_PATH}")

print("\nGPU SUMMARY")

print(f"GPU: {torch.cuda.get_device_name(0)}")

print(
    "Peak GPU Memory: "
    f"{torch.cuda.max_memory_allocated(0) / (1024 ** 3):.2f} GB"
)
