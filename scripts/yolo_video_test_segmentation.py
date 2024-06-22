from ultralytics import YOLO
import cv2
import os
from os.path import join
import cvzone
import math

classNames = [
    'bottle'   
]

#model = YOLO(r'C:\Users\amanda\Desktop\Yolo\runs\detect\yolov8n_full_lenght5\weights\best.pt')
model = YOLO(r'/home/mandanamikata/Desktop/BKP_AMANDA/Yolo/runs/segment/yolov8n_seg_vol_hue_saturation-good/weights/best.pt')
model.predict(
    
    source="/home/mandanamikata/Desktop/BKP_AMANDA/Yolo/2,0COCACOLA ZERO FSC.mp4",
    show=True,
    conf=0.5,   
    
    )  
   
# video = r''

# cap = cv2.VideoCapture(0)
# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))

# while True:
#     success, img = cap.read()
#     if not success:
#         break
#     results = model(img, stream=True)

#     # Process each detected object
#     for r in results:
#         boxes = r.boxes
#         for box in boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             w, h = x2 - x1, y2 - y1
#             conf = math.ceil(box.conf[0] * 100) / 100
#             cls = int(box.cls[0])

#             if conf > 0.55:
#                 cvzone.cornerRect(img, (x1, y1, w, h))
#                 cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=2, thickness=2)

#     # Display the resulting frame
#     cv2.imshow('Image', img)

#     # Break the loop on 'q' key press
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the video capture object
# cap.release()

# # Close all OpenCV windows
# cv2.destroyAllWindows()
