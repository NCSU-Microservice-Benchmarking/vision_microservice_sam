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
            car_indices = torch.where((clss == 2) | (clss == 5) | (clss == 8))[0]  # [0] to extract tensor indices
            # Use these indices to extract the relevant masks
            car_masks = masks[car_indices]
    
            for i in range(len(car_masks)):
                # Crop the individual object from the frame
                instance_bbox = bounding_boxes[car_indices[i]].cpu().numpy()
                # Crop the object from the frame
                mask = car_masks[i]
            
                print(f"Mask {i} shape: {mask.shape}")
                # Scale for visualizing results
                var = mask.byte() * 1
                # will need to check the proper indices
                pixels = np.sum(var) /( img.shape[0] * img.shape[1])
                if pixels > 0.5:
                    people_mask = mask.byte() * 255  # Convert to uint8 before multiplying by 255
                    cv2.imwrite(str(model.predictor.save_dir) + img_path + str + '.png', people_mask.cpu().numpy())
                
                # people_mask = mask.byte() * 255  # Convert to uint8 before multiplying by 255
                # print(f"Scaled mask {i} shape: {people_mask.shape}")
                # Save to file
                # cv2.imwrite(str(model.predictor.save_dir) +'/merged_segs' + str(i) + '.png', people_mask.cpu().numpy())
    except Exception as e:
        print(f"An error occurred while processing {img_path}: {e}")