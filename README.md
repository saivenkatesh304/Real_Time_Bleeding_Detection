# Robotic Surgery Bleeding Detection

This project implements an **AI-based alert system** designed to detect **intraoperative bleeding** during robotic surgery using **semantic segmentation** techniques. The goal is to improve patient safety by enabling timely detection of bleeding events during surgical procedures.

## Features
- **Semantic Segmentation** using a UNet architecture with a ResNet34 encoder.
- Trained on surgical video frames and corresponding bleeding masks.
- Tracks performance metrics such as Precision, Recall, F1-Score, IoU, Accuracy, and Specificity.
- Supports evaluation on test datasets with detailed metric reports and visualizations.

## Tech Stack
- **Language:** Python
- **Framework:** PyTorch
- **Model:** UNet with ResNet34 encoder
- **Dataset:** Surgical video frames (640x640) with bleeding masks
- **Tools:** Roboflow (annotation), OpenCV, Matplotlib, Pandas

- ## Workflow
**Dataset Preparation:** Extract frames from surgical videos, annotate bleeding regions, and split into train/validation/test sets
**Model Training:** Train UNet with ResNet34 encoder on the prepared dataset
**Evaluation:** Assess model performance using quantitative metrics and visual results
**Deployment:** Implement real-time inference for surgical use cases




For details on dataset contact saiv14076@gmail.com**
