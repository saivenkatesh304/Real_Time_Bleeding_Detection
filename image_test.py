import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import segmentation_models_pytorch as smp
from sklearn.metrics import precision_score, recall_score, jaccard_score, f1_score, accuracy_score, confusion_matrix
import csv

# Paths
test_images_dir = r"C:\Users\Sai Venkatesh\Desktop\Final Project\test\images"
test_masks_dir = r"C:\Users\Sai Venkatesh\Desktop\Final Project\test\masks"
output_dir = r"C:\Users\Sai Venkatesh\Desktop\Final Project\test\comparison_outputs"
os.makedirs(output_dir, exist_ok=True)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model
model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Store predictions and targets for metrics
all_preds = []
all_targets = []

# Loop through test images
for img_name in os.listdir(test_images_dir):
    if not img_name.lower().endswith(('.jpg', '.png')):
        continue

    img_path = os.path.join(test_images_dir, img_name)

    # Correct mask filename logic:
    mask_name = img_name.rsplit('.', 1)[0] + "_mask.png"
    mask_path = os.path.join(test_masks_dir, mask_name)

    if not os.path.exists(mask_path):
        print(f"Mask not found for {img_name}, skipping.")
        continue

    # Load image and mask
    image = Image.open(img_path).convert("RGB")
    gt_mask = Image.open(mask_path).convert("L")

    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.sigmoid(output)
        pred_mask = (probs > 0.2).float().squeeze().cpu().numpy()

    gt_mask_np = np.array(gt_mask.resize((256, 256), Image.NEAREST))
    pred_mask_bin = (pred_mask > 0.5).astype(np.uint8)
    gt_mask_bin = (gt_mask_np > 0).astype(np.uint8)

    all_preds.extend(pred_mask_bin.flatten())
    all_targets.extend(gt_mask_bin.flatten())

    # Create comparison image with original, predicted mask, and overlay with outline
    image_np = np.array(image.resize((256, 256)))

    # Predicted mask grayscale
    pred_mask_gray = (pred_mask_bin * 255).astype(np.uint8)
    pred_mask_bgr = cv2.cvtColor(pred_mask_gray, cv2.COLOR_GRAY2BGR)

    # Overlay: original + predicted mask contour in red
    overlay = image_np.copy()
    contours, _ = cv2.findContours(pred_mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)  # Red contours

    combined = np.hstack([
        image_np,          # Original image
        pred_mask_bgr,     # Predicted mask grayscale
        overlay            # Original with mask outline
    ])

    out_img = Image.fromarray(combined)
    out_img.save(os.path.join(output_dir, f"compare_{img_name}"))

# Compute metrics if any predictions were made
if len(all_preds) == 0 or len(all_targets) == 0:
    print("No valid predictions or targets found. Metrics computation skipped.")
else:
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    iou = jaccard_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    accuracy = accuracy_score(all_targets, all_preds)
    tn, fp, fn, tp = confusion_matrix(all_targets, all_preds).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Save metrics
    metrics = {
        "Precision": precision,
        "Recall": recall,
        "IoU": iou,
        "F1 Score": f1,
        "Accuracy": accuracy,
        "Specificity": specificity,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn
    }

    with open("test_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        for k, v in metrics.items():
            writer.writerow([k, v])

    print("✅ Comparison images saved to:", output_dir)
    print("📄 Test metrics saved to: test_metrics.csv")
