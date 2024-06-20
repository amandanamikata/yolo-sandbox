from ultralytics import YOLO
import cv2
import numpy as np
import math

# Load a model
model = YOLO('yolov8n-seg.yaml')  # build a new model from YAML
model = YOLO('yolov8n-seg.pt')  # load a pretrained model (recommended for training)
model = YOLO('yolov8n-seg.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model
#results = model.train(data='coco8-seg.yaml', epochs=100, imgsz=640)

#-------------------------------------------------VIDEO

img = r"C:\Users\amanda\Desktop\Yolo\dataset\doggo2.jpg"

results = model.predict(img)#, show=True)
print(results)

for r in results:
    masks = r.masks    
    for mask in masks:
        
        print(f"Mask --------------- {mask} -----------")
        
        mask_data = mask.numpy() * 255  # Convert mask to uint8 format

        # Display the mask using OpenCV
        cv2.imshow("Mask", mask_data.astype("uint8"))
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
