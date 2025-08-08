import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import segmentation_models_pytorch as smp

def segment_video(model, video_path, output_path, device='cuda'):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30, (frame_width, frame_height))

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model.eval()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # Removed manual resize, transform does it

        input_tensor = transform(input_image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            prob_map = torch.sigmoid(output).squeeze().cpu().numpy()

        max_prob = prob_map.max()
        if max_prob < 0.3:  # confidence threshold
            out.write(frame)
            continue

        mask = (prob_map > 0.5).astype(np.uint8)
        mask = cv2.resize(mask, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = 500  # Adjust this threshold as needed
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

        cv2.drawContours(frame, filtered_contours, -1, (255, 0, 0), 2)  # blue outlines

        out.write(frame)

    cap.release()
    out.release()
    print(f"Video saved to {output_path}")


# Load your model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1).to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))

# Call the function
segment_video(model, 'test4.mp4', 'output_video_with_outlines4.mp4', device=device)
