import cv2
from ultralytics import YOLO
import numpy as np
import torch


img= cv2.imread('bus.png')
model = YOLO('yolov8m-seg.pt')
results = model.predict(source=img.copy(), save=True, save_txt=False, stream=True)
for result in results:
    # get array results
    masks = result.masks.data
    boxes = result.boxes.data
    # extract classes
    clss = boxes[:, 5]
    # get indices of results where class is 0 (people in COCO)
    people_indices = torch.where(clss == 2)
    # use these indices to extract the relevant masks
    people_masks = masks[people_indices]
    
    for i in range(len(people_masks)):
        mask = people_masks[i]
        print(mask.shape)
        # scale for visualizing results
        people_mask = mask.int() * 255
        print(people_mask.shape)
        # save to file
        cv2.imwrite(str(model.predictor.save_dir) +'/merged_segs' + str(i) + '.jpg', people_mask.cpu().numpy())