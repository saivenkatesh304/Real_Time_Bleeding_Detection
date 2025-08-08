import os
import csv
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, jaccard_score, f1_score, accuracy_score, confusion_matrix

# 1. Dataset Class
class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, target_size=(256, 256)):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.target_size = target_size
        self.images = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        image_path = os.path.join(self.images_dir, image_name)
        mask_path = os.path.join(self.masks_dir, image_name.replace('.jpg', '_mask.png'))

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = image.resize(self.target_size, Image.BILINEAR)
        mask = mask.resize(self.target_size, Image.NEAREST)

        if self.transform:
            image = self.transform(image)

        mask = torch.from_numpy(np.array(mask)).float()
        mask = (mask > 0).float().unsqueeze(0)

        return image, mask

# 2. Data loaders
def get_loaders():
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_dataset = SegmentationDataset(
        r"C:\Users\Sai Venkatesh\Desktop\Final Project\train\images",
        r"C:\Users\Sai Venkatesh\Desktop\Final Project\train\masks",
        transform
    )
    valid_dataset = SegmentationDataset(
        r"C:\Users\Sai Venkatesh\Desktop\Final Project\valid\images",
        r"C:\Users\Sai Venkatesh\Desktop\Final Project\valid\masks",
        transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)
    return train_loader, valid_loader

# 3. Model
def get_model():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    return model

# 4. Loss
def get_criterion(device):
    pos_weight = torch.tensor([50.0]).to(device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# 5. Evaluation
def evaluate(model, data_loader, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, masks in data_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            threshold = 0.2 * probs.max().item()
            preds = (probs > threshold).float()
            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(masks.cpu().numpy().flatten())

    if np.sum(all_preds) == 0:
        print("Warning: No predictions above threshold")
        return {
            'precision': 0.0,
            'recall': 0.0,
            'iou': 0.0,
            'f1': 0.0,
            'accuracy': 0.0,
            'specificity': 0.0,
            'tp': 0,
            'fp': 0,
            'tn': 0,
            'fn': 0
        }

    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    iou = jaccard_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    accuracy = accuracy_score(all_targets, all_preds)

    # Confusion matrix counts
    tn, fp, fn, tp = confusion_matrix(all_targets, all_preds).ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'iou': iou,
        'f1': f1,
        'accuracy': accuracy,
        'specificity': specificity,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn
    }

# 6. Training loop with metric saving and plotting
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_loader, valid_loader = get_loaders()
    model = get_model().to(device)
    criterion = get_criterion(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_iou = 0.0

    # Lists to store metrics for plotting and saving
    train_losses = []
    val_precisions = []
    val_recalls = []
    val_ious = []
    val_f1s = []
    val_accuracies = []
    val_specificities = []

    for epoch in range(50):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        metrics = evaluate(model, valid_loader, device)

        train_losses.append(avg_train_loss)
        val_precisions.append(metrics['precision'])
        val_recalls.append(metrics['recall'])
        val_ious.append(metrics['iou'])
        val_f1s.append(metrics['f1'])
        val_accuracies.append(metrics['accuracy'])
        val_specificities.append(metrics['specificity'])

        print(f"Epoch {epoch+1}: Loss={avg_train_loss:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, IoU={metrics['iou']:.4f}, F1={metrics['f1']:.4f}, Accuracy={metrics['accuracy']:.4f}, Specificity={metrics['specificity']:.4f}")

        if metrics['iou'] > best_iou:
            best_iou = metrics['iou']
            torch.save(model.state_dict(), "best_model.pth")

    # Save metrics to CSV
    with open("training_metrics.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Loss', 'Precision', 'Recall', 'IoU', 'F1 Score', 'Accuracy', 'Specificity'])
        for i in range(len(train_losses)):
            writer.writerow([i+1, train_losses[i], val_precisions[i], val_recalls[i], val_ious[i], val_f1s[i], val_accuracies[i], val_specificities[i]])
    print("Training metrics saved to training_metrics.csv")

    epochs = range(1, len(train_losses) + 1)

    def save_plot(x, y, xlabel, ylabel, title, filename):
        plt.figure()
        plt.plot(x, y, marker='o')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.savefig(filename)
        plt.close()

    save_plot(epochs, train_losses, 'Epoch', 'Loss', 'Training Loss', 'train_loss.png')
    save_plot(epochs, val_precisions, 'Epoch', 'Precision', 'Validation Precision', 'val_precision.png')
    save_plot(epochs, val_recalls, 'Epoch', 'Recall', 'Validation Recall', 'val_recall.png')
    save_plot(epochs, val_ious, 'Epoch', 'IoU', 'Validation IoU', 'val_iou.png')
    save_plot(epochs, val_f1s, 'Epoch', 'F1 Score', 'Validation F1 Score', 'val_f1.png')
    save_plot(epochs, val_accuracies, 'Epoch', 'Accuracy', 'Validation Accuracy', 'val_accuracy.png')
    save_plot(epochs, val_specificities, 'Epoch', 'Specificity', 'Validation Specificity', 'val_specificity.png')
    print("Individual metric plots saved.")

    # Combined plot
    plt.figure(figsize=(14, 10))

    plt.subplot(3, 3, 1)
    plt.plot(epochs, train_losses, marker='o')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.subplot(3, 3, 2)
    plt.plot(epochs, val_precisions, marker='o', color='orange')
    plt.title('Validation Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.grid(True)

    plt.subplot(3, 3, 3)
    plt.plot(epochs, val_recalls, marker='o', color='green')
    plt.title('Validation Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.grid(True)

    plt.subplot(3, 3, 4)
    plt.plot(epochs, val_ious, marker='o', color='red')
    plt.title('Validation IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.grid(True)

    plt.subplot(3, 3, 5)
    plt.plot(epochs, val_f1s, marker='o', color='purple')
    plt.title('Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.grid(True)

    plt.subplot(3, 3, 6)
    plt.plot(epochs, val_accuracies, marker='o', color='brown')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)

    plt.subplot(3, 3, 7)
    plt.plot(epochs, val_specificities, marker='o', color='cyan')
    plt.title('Validation Specificity')
    plt.xlabel('Epoch')
    plt.ylabel('Specificity')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('combined_metrics.png')
    plt.close()
    print("Combined metric plot saved.")

if __name__ == "__main__":
    train()
