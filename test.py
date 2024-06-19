import cv2
from ultralytics import YOLO
import numpy as np
import torch
import os

# Change this to the correct path
input_dir = r'train'

# Initialize the model once, outside the loop
model = YOLO('yolov8m-seg.pt')
model.predict(source="0", show=True, stream=True, classes=[2, 6, 7])  # [0, 3, 5] for multiple classes

# Ensure the output directory exists
output_dir = os.path.join('merged_segs')
os.makedirs(output_dir, exist_ok=True)

# Loop through all images in the directory
for img_name in os.listdir(input_dir):
    img_path = os.path.join(input_dir, img_name)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to read image: {img_path}")
        continue

    try: 
        results = model.predict(source=img.copy(), save=True, save_txt=False, stream=True)
        
        for result in results:
         # Check if any masks were detected
            if result.masks is None:
                print(f"No detections in image: {img_path}")
                continue

            # Get array results
            masks = result.masks.data
            boxes = result.boxes
            # Extract classes
            clss = boxes.data[:, 5]
            bounding_boxes = boxes.xyxy    
            # Get indices of results where class is 0 (people in COCO)
            car_indices = torch.where((clss == 2) | (clss == 6) | (clss == 7))[0]  # [0] to extract tensor indices
             # Calculate areas of bounding boxes
            areas = (bounding_boxes[car_indices][:, 2] - bounding_boxes[car_indices][:, 0]) * \
                    (bounding_boxes[car_indices][:, 3] - bounding_boxes[car_indices][:, 1])

            # Get the index of the largest bounding box
            largest_idx = car_indices[torch.argmax(areas)]

            # Extract the relevant mask for the largest vehicle
            largest_mask = masks[largest_idx]
            largest_bbox = bounding_boxes[largest_idx].cpu().numpy()

            # Process the mask of the largest vehicle
            print(f"Largest mask shape: {largest_mask.shape}")

            # Scale for visualizing results
            vehicle_mask = largest_mask.byte() * 255  # Convert to uint8 before multiplying by 255
            print(f"Scaled mask shape: {vehicle_mask.shape}")

            # Save to file
            save_path = os.path.join(output_dir, f"{img_name}_largest_vehicle.png")
            cv2.imwrite(save_path, vehicle_mask.cpu().numpy())
            print(f"Saved largest vehicle mask to: {save_path}")
    except Exception as e:
        print(f"An error occurred while processing {img_path}: {e}")