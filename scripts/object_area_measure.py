from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np
import math

def getContours(img, imgContour, conf):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if conf > 0.55:
                    
            cv2.drawContours(imgContour, [cnt], -1, (255, 0, 255), 4)                  
            cv2.putText(imgContour, "conf: "+str(conf), (0, 20), cv2.FONT_HERSHEY_DUPLEX, .7, (100, 100, 100),2)
            cv2.putText(imgContour, "area: "+str(int(area)), (0, 45), cv2.FONT_HERSHEY_DUPLEX, 0.7, (100, 100, 100), 2)
    
    return imgContour

model = YOLO(r'/home/mandanamikata/Desktop/BKP_AMANDA/Yolo/runs/segment/yolov8n_seg_vol_hue_saturation-good/weights/best.pt')

frameWidth = 640
frameHeight = 480
videopath = r"/home/mandanamikata/Desktop/BKP_AMANDA/Yolo/2,0COCACOLA ZERO FSC.mp4"
cap = cv2.VideoCapture(0)  
cap.set(3, frameWidth)
cap.set(4, frameHeight)

while True:
    success, img = cap.read()
    if not success:
        break    
    imgContour = img.copy()
    res = model(img, stream=True)

    # Iterate detection results
    for r in res:
        img = np.copy(r.orig_img)
        img_name = Path(r.path).stem

        # Iterate each object contour
        for ci, c in enumerate(r):
            label = c.names[c.boxes.cls.tolist().pop()]          
            boxes = r.boxes            
            for box in boxes:
                conf = math.ceil(box.conf[0] * 100) / 100          

            b_mask = np.zeros(img.shape[:2], np.uint8)
            
            # Create contour mask
            contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
            _ = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)           

            # OPTION-1: Isolate object with black background
            mask3ch = cv2.cvtColor(b_mask, cv2.COLOR_GRAY2BGR)
            isolated = cv2.bitwise_and(mask3ch, img)           

            # OPTION-2: Isolate object with transparent background (when saved as PNG)
            isolated2 = np.dstack([img, b_mask])            

            # OPTIONAL: detection crop (from either OPT1 or OPT2)
            x1, y1, x2, y2 = c.boxes.xyxy.cpu().numpy().squeeze().astype(np.int32)
            iso_crop = isolated[y1:y2, x1:x2]          
            imgContour = getContours(b_mask, imgContour, conf)

        cv2.imshow("Result", imgContour)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
