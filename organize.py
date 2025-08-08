import os
import shutil

# Set your base directory
base_dir = r"C:\Users\Sai Venkatesh\Desktop\Final Project"

# Folders to organize
for folder_name in ['train', 'valid', 'test']:
    folder_path = os.path.join(base_dir, folder_name)
    
    # Make images/ and masks/ subfolders
    os.makedirs(os.path.join(folder_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(folder_path, 'masks'), exist_ok=True)

    # Move files
    for file in os.listdir(folder_path):
        full_path = os.path.join(folder_path, file)
        if os.path.isfile(full_path):
            if '_mask' in file:
                shutil.move(full_path, os.path.join(folder_path, 'masks', file))
            elif file.endswith(('.jpg', '.png')):  # image files
                shutil.move(full_path, os.path.join(folder_path, 'images', file))
            # Leave _classes.csv untouched

print("Train, Valid, and Test folders are now organized into images, masks, and CSV.")
