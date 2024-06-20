from ultralytics import YOLO
import cv2
import os
from os.path import join
import cvzone
import math

classNames = [
    'bottle',        
]

#model = YOLO(r'C:\Users\amanda\Desktop\Yolo\runs\detect\yolov8n_full_lenght5\weights\best.pt')
model = YOLO(r'/home/mandanamikata/Desktop/BKP_AMANDA/Yolo/runs/segment/yolov8n_seg_vol_hue_saturation-good/weights/best.pt')
#model.predict(source="0", show=True, conf=0.5)  
   
video = r''

cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

while True:
    success, img = cap.read()
    if not success:
        break
    results = model(img, stream=True)
    
    # Process each detected object
    for r in results:
        print(r)
        #r.show()
        boxes = r.boxes
        print(boxes)
        for box in boxes:
            print(box)
            x1, y1, x2, y2 = map(int, box.xyxy[0])            
           # w, h = x2 - x1, y2 - y1
            #conf = math.ceil(box.conf[0] * 100) / 100
            #cls = int(box.cls[0])
            
            #cvzone.cornerRect(img, (x1, y1, w, h))
            #cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=2, thickness=2)

    # Display the resulting frame
    cv2.imshow('Image', img)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
 
 #--------------------------
 
# height, width, channels= img.shape

# results = model.predict(source="0", save=False, save_txt=False)
# result = results[0]
# segmentation_contours_idx= []

# for seg in result.masks.segments:
#     seg[:,0] *= width
#     seg[:,1] *= height
#     segment = np.array(seg, dtype=np.int32)
#     segmentation_contours_idx.append(segment)
    
# bboxes = np.array(result.boxes.xyxy.cpu(),dtype="int")

# class_ids = np.array(result.boxes.cls.cpu(),dtype="int")

# scores = np.array(result.boxes.conf.cpu(),dtype="float").round(2)
