import os
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from segmentation_models_pytorch import Unet
from sklearn.metrics import precision_score, recall_score, jaccard_score, f1_score, accuracy_score, confusion_matrix

# Dataset class (same as training)
class SegmentationDataset(torch.utils.data.Dataset):
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

# Load model architecture
def get_model():
    model = Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    return model

# Evaluate function
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

def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Test data transform
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Paths to test images and masks - update if needed
    test_images_dir = r"C:\Users\Sai Venkatesh\Desktop\Final Project\test\images"
    test_masks_dir = r"C:\Users\Sai Venkatesh\Desktop\Final Project\test\masks"

    test_dataset = SegmentationDataset(test_images_dir, test_masks_dir, transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Load model and weights
    model = get_model().to(device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))

    # Evaluate
    test_metrics = evaluate(model, test_loader, device)

    # Print results in table format
    print("\nTest Results:")
    print("-" * 40)
    print(f"{'Metric':<15} | {'Value':<10}")
    print("-" * 40)
    for key, value in test_metrics.items():
        if key in ['tp', 'fp', 'tn', 'fn']:
            continue  # Skip raw confusion matrix counts here
        print(f"{key.capitalize():<15} | {value:.4f}")
    print("-" * 40)

    # Save results to CSV
    with open("test_results.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Metric', 'Value'])
        for key, value in test_metrics.items():
            writer.writerow([key, value])
    print("Test metrics saved to test_results.csv")

if __name__ == "__main__":
    test()
