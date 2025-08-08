import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report
import segmentation_models_pytorch as smp

# Dataset
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

# Load model
def get_model():
    return smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)

# Main function
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset = SegmentationDataset(
        r"C:\Users\Sai Venkatesh\Desktop\Final Project\test\images",
        r"C:\Users\Sai Venkatesh\Desktop\Final Project\test\masks",
        transform
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    model = get_model().to(device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            threshold = 0.2 * probs.max().item()
            preds = (probs > threshold).float()

            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(masks.cpu().numpy().flatten())

    all_preds = np.array(all_preds) > 0.5
    all_targets = np.array(all_targets) > 0.5

    report = classification_report(all_targets, all_preds, target_names=["Background", "Bleeding"], digits=2)
    print("\nClassification Report:\n")
    print(report)

    with open("classification_report.txt", "w") as f:
        f.write("Classification Report\n\n")
        f.write(report)
    print("Classification report saved to classification_report.txt")

if __name__ == "__main__":
    main()
