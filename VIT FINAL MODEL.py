# ---------------------------------------------------
# Vision Capsule Project: ViT Model for Classification
# ---------------------------------------------------
# This project trains a Vision Transformer (ViT) model 
# to classify medical abnormalities in video capsule endoscopy.
# Author: ROSHON R, SHUSHMITA K
# ---------------------------------------------------


# -----------------------------
# 1. Import Required Libraries
# -----------------------------
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import ViTForImageClassification
import torch.optim as optim
from tqdm import tqdm  # For progress bars
from collections import Counter
import numpy as np
import pandas as pd
import shutil
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc, 
    f1_score, balanced_accuracy_score
)


# -----------------------------
# 2. Device Configuration
# -----------------------------
# Use GPU if available; otherwise, use CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -----------------------------
# 3. Define Class Names
# -----------------------------
class_names = [
    'Angioectasia', 'Bleeding', 'Erosion', 'Erythema', 'Foreign Body',
    'Lymphangiectasia', 'Polyp', 'Ulcer', 'Worms', 'Normal'
]


# -----------------------------
# 4. Define Dataset Paths
# -----------------------------
train_path = r'D:\projects college\Dataset\Pre-Processed Dataset\train' #define your own path
val_path = r'D:\projects college\Dataset\Pre-Processed Dataset\validation'  #define your own path
test_images_dir = r'D:\projects college\Dataset\Pre-Processed Dataset\test'  #define your own path


# -----------------------------
# 5. Define Image Transformations
# -----------------------------
# Resize images to 224x224, convert to tensor, and normalize.
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


# -----------------------------
# 6. Load Datasets Using ImageFolder
# -----------------------------
train_dataset = datasets.ImageFolder(train_path, transform=transform)
val_dataset = datasets.ImageFolder(val_path, transform=transform)


# -----------------------------
# 7. Create a Balanced DataLoader
# -----------------------------
def create_balanced_dataloader(dataset, batch_size=32):
    """Create a DataLoader with class-balanced sampling."""
    class_counts = Counter([label for _, label in dataset.samples])
    class_weights = {cls: len(dataset) / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[label] for _, label in dataset.samples]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=2, pin_memory=True)

# Initialize DataLoaders
train_loader = create_balanced_dataloader(train_dataset)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)


# -----------------------------
# 8. Load Pre-trained ViT Model
# -----------------------------
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224-in21k',
    num_labels=len(class_names)
)
model.to(device)


# -----------------------------
# 9. Define Optimizer, Loss, and Scheduler
# -----------------------------
optimizer = optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


# -----------------------------
# 10. Define Training and Validation Functions
# -----------------------------
def train_one_epoch(model, data_loader, optimizer, criterion):
    """Train the model for one epoch."""
    model.train()
    total_loss, total_correct = 0, 0

    for images, labels in tqdm(data_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += (outputs.argmax(1) == labels).sum().item()

    return total_loss / len(data_loader), total_correct / len(data_loader.dataset)

def validate(model, data_loader, criterion):
    """Validate the model."""
    model.eval()
    total_loss, total_correct = 0, 0

    with torch.no_grad():
        for images, labels in tqdm(data_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            total_correct += (outputs.argmax(1) == labels).sum().item()

    return total_loss / len(data_loader), total_correct / len(data_loader.dataset)


# -----------------------------
# 11. Training Loop
# -----------------------------
num_epochs = 10

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, criterion)
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

    val_loss, val_accuracy = validate(model, val_loader, criterion)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    scheduler.step()

print("Training complete!")


# -----------------------------
# 12. Save the Trained Model
# -----------------------------
model_save_path = r'C:\Users\XEON\Documents\trainer for VIT\VIT FINAL\whole-he_vit_model.pth'   #define your own path
torch.save(model.state_dict(), model_save_path)
print(f'Model saved to {model_save_path}')




# ------------------------------
# 13.Device Setup
# ------------------------------
# Check if a GPU is available; if not, use the CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Move the model to the selected device and switch to evaluation mode
model.eval()
model.to(device)


# ------------------------------
# 14. Define Class Names For Evaluation
# ------------------------------
class_names = [
    'Angioectasia', 'Bleeding', 'Erosion', 'Erythema', 'Foreign Body',
    'Lymphangiectasia', 'Polyp', 'Ulcer', 'Worms', 'Normal'
]


# ------------------------------
# 15.Function to Get Predictions, Labels, and Probabilities
# ------------------------------
def get_predictions(model, data_loader):
    """Run inference on the data loader and collect predictions, labels, and probabilities."""
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass through the model
            outputs = model(images).logits  # Assuming output is logits
            probs = torch.softmax(outputs, dim=1)  # Get probabilities

            # Collect predictions, true labels, and probabilities
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


# ------------------------------
# 16.Generate Predictions and Labels from Validation Data
# ------------------------------
val_preds, val_labels, val_probs = get_predictions(model, val_loader)

# Generate a classification report
print("Classification Report:")
print(classification_report(val_labels, val_preds, target_names=class_names))


# ------------------------------
# 17.Calculate Sensitivity, Specificity, and Related Metrics
# ------------------------------
def calculate_metrics(y_true, y_pred, num_classes):
    """Calculate sensitivity, specificity, and other metrics."""
    conf_matrix = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)

    # Calculate sensitivity (recall) for each class
    sensitivity = [report[str(i)]['recall'] for i in range(num_classes)]
    mean_sensitivity = np.mean(sensitivity)

    # Calculate specificity for each class
    specificity = []
    for i in range(num_classes):
        TN = conf_matrix.sum() - (conf_matrix[i].sum() + conf_matrix[:, i].sum() - conf_matrix[i, i])
        FP = conf_matrix[:, i].sum() - conf_matrix[i, i]
        specificity.append(TN / (TN + FP))

    mean_specificity = np.mean(specificity)

    return sensitivity, mean_sensitivity, specificity, mean_specificity

# Calculate sensitivity, specificity, and other metrics
sensitivity, mean_sensitivity, specificity, mean_specificity = calculate_metrics(
    val_labels, val_preds, len(class_names)
)
balanced_accuracy = balanced_accuracy_score(val_labels, val_preds)
weighted_f1 = f1_score(val_labels, val_preds, average='weighted')

# Print out the calculated metrics
print(f'Sensitivity per class: {sensitivity}')
print(f'Mean Sensitivity: {mean_sensitivity:.4f}')
print(f'Specificity per class: {specificity}')
print(f'Mean Specificity: {mean_specificity:.4f}')
print(f'Balanced Accuracy: {balanced_accuracy:.4f}')
print(f'Weighted F1 Score: {weighted_f1:.4f}')


# ------------------------------
# 18.Plot Normalized Confusion Matrix
# ------------------------------
conf_matrix = confusion_matrix(val_labels, val_preds)
conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(10, 8))
sns.heatmap(
    conf_matrix_normalized, annot=True, cmap='Blues',
    xticklabels=class_names, yticklabels=class_names
)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Normalized Confusion Matrix for Validation Set')
plt.show()


# ------------------------------
# 19.Plot AUC-ROC Curve for Each Class
# ------------------------------
val_labels_one_hot = np.eye(len(class_names))[val_labels]  # Convert labels to one-hot encoding
roc_aucs = []

# Calculate AUC-ROC for each class
for i in range(len(class_names)):
    fpr, tpr, _ = roc_curve(val_labels_one_hot[:, i], val_probs[:, i])
    roc_auc = auc(fpr, tpr)
    roc_aucs.append(roc_auc)

    # Plot each class's ROC curve
    plt.plot(fpr, tpr, label=f'Class {class_names[i]} (AUC = {roc_auc:.2f})')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AUC-ROC for Validation Set')
plt.legend(loc='best')
plt.show()

# Calculate and print the average AUC-ROC score
avg_auc_roc = np.mean(roc_aucs)
print(f'Average AUC-ROC: {avg_auc_roc:.4f}')


# ------------------------------
# 20.Preprocessing Function for Test Images
# ------------------------------
def preprocess_image(image):
    """Preprocess an image for the model."""
    image = cv2.resize(image, (224, 224))  # Resize to the input size
    image = image / 255.0  # Normalize pixel values to [0, 1]
    image = np.transpose(image, (2, 0, 1))  # Change shape to (C, H, W)
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    return image


# ------------------------------
# 21.Directory Paths and Excel Export Setup
# ------------------------------
test_images_dir = r'C:\Users\XEON\Documents\trainer for VIT\dataset\test'  #define your own path
output_excel_path = r'C:\Users\XEON\Documents\trainer for VIT\VIT FINAL\check_vit_test_predictions.xlsx'   #define your own path


# ------------------------------
# 22.Function to Save Predictions to Excel
# ------------------------------
def save_predictions_to_excel(output_excel_path):
    """Run inference on test images and save predictions to an Excel file."""
    data = []  # Store image names, probabilities, and predictions

    with torch.no_grad():
        for img_name in os.listdir(test_images_dir):
            img_path = os.path.join(test_images_dir, img_name)
            image = preprocess_image(cv2.imread(img_path)).to(device)

            # Get model predictions and probabilities
            outputs = model(image).logits
            probs = torch.softmax(outputs, dim=1).cpu().numpy().flatten()

            # Store results for each image
            pred_class = class_names[np.argmax(probs)]  # Get predicted class
            row = [img_name] + probs.tolist() + [pred_class]  # Save data as a row
            data.append(row)

    # Create a DataFrame with columns for image paths, probabilities, and predictions
    columns = ['Image Path'] + class_names + ['Predicted Class']
    df = pd.DataFrame(data, columns=columns)

    # Save the DataFrame to an Excel file
    df.to_excel(output_excel_path, index=False)
    print(f"Test predictions saved to {output_excel_path}")


# ------------------------------
# 23.Save Test Predictions to Excel
# ------------------------------
save_predictions_to_excel(output_excel_path)


# -----------------------------
# 24. Classify Test Images and Save Results
# -----------------------------
classified_output_dir = r'C:\Users\XEON\Documents\trainer for VIT\VIT FINAL'   #define your own path

for class_name in class_names:
    os.makedirs(os.path.join(classified_output_dir, class_name), exist_ok=True)

def preprocess_image(image):
    """Preprocess images for model input."""
    image = cv2.resize(image, (224, 224))
    image = torch.tensor(image).float().permute(2, 0, 1).unsqueeze(0)
    image = (image / 255.0 - 0.5) / 0.5
    return image

def classify_and_save_images():
    """Classify and save test images."""
    model.eval()
    with torch.no_grad():
        for img_name in os.listdir(test_images_dir):
            img_path = os.path.join(test_images_dir, img_name)
            image = preprocess_image(cv2.imread(img_path)).to(device)
            predictions = model(image).logits
            predicted_class = class_names[torch.argmax(predictions).item()]
            shutil.copy(img_path, os.path.join(classified_output_dir, predicted_class, img_name))

classify_and_save_images()
print(f'Test data classified and saved to {classified_output_dir}')