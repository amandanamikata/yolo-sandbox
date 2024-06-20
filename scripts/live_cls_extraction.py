from ultralytics import YOLO
import cv2
import math
from os.path import join

#from src.services.machine_learning.cap_detector_service import cap_detector_service

classNames = ['bottom', 'code', 'label', 'neck']

def crop_frame(frame, bbox, target_size=(500, 500), padding=20):
    x_min, y_min, x_max, y_max = bbox
    # Calculate the center of the bounding box
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    # Calculate half the target size
    half_width = target_size[0] // 2
    half_height = target_size[1] // 2
    # Calculate the new bounding box coordinates
    x_min_pad = max(0, int(center_x - half_width) - padding)
    y_min_pad = max(0, int(center_y - half_height) - padding)
    x_max_pad = min(frame.shape[1], int(center_x + half_width) + padding)
    y_max_pad = min(frame.shape[0], int(center_y + half_height) + padding)
    # Crop the frame
    cropped_frame = frame[y_min_pad:y_max_pad, x_min_pad:x_max_pad]
    # Resize the cropped frame to the target size
    cropped_frame = cv2.resize(cropped_frame, target_size)
    return cropped_frame

def detect_and_crop(video_path, model_path, output_dir, confidence):
    # Load the YOLO model
    model = YOLO(model_path)

    # Open the video capture object
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    while True:
        success, img = cap.read()
        if not success:
            break      
        results = model(img, stream=False)
        for r in results:
            boxes = r.boxes
            for box in boxes:       
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1 
                conf = math.ceil(box.conf[0] * 100) / 100
                cls = int(box.cls[0])     

                if cls == 3 and conf > confidence:                                    
                    x1, y1, x2, y2 = box.xyxy[0]     
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)                
                    cropped_frame = crop_frame(img, bbox=(x1, y1, x2, y2), padding=40)

                    # Save the cropped frame
                    output_path = join(output_dir, 'cropped_frame.jpg')
                    cv2.imwrite(output_path, cropped_frame)

                    # Display the cropped frame
                    
                    #print(cap_detector_service(cropped_frame))
                    cv2.imshow("Cropped Frame", cropped_frame)
                    cv2.waitKey(0)  # Wait indefinitely until a key is pressed
                    cv2.destroyAllWindows()  # Close the window
                    
  # Release the video capture object
    cap.release()
    
    return cropped_frame

# Paths and parameters
video_path = r'C:\Users\amanda\Desktop\Yolo\videos' 
model_path = r'C:\Users\amanda\Desktop\Yolo\runs\detect\yolov8n_custom3_final\weights\best.pt'
output_dir = r'C:\Users\amanda\Desktop\Yolo\output_videos' 

# Call the function to detect and crop
detect_and_crop(video_path, model_path, output_dir, 0.90)
