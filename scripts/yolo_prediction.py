from ultralytics import YOLO
import cv2
import os

# Initialize YOLO model
model = YOLO(r'C:\Users\amanda\Desktop\Yolo\runs\detect\yolov8n_full_lenght5\weights\best.pt')

# Directory containing images
image_dir = r'C:\Users\amanda\Desktop\Yolo\dataset\random_internet_imgs'

# Maximum height of the window
max_height = 800

# Loop through each image in the directory
for filename in os.listdir(image_dir):
    if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg') or filename.endswith('.webp'):

        # Construct full path to the image
        image_path = os.path.join(image_dir, filename)

        # Process the image using YOLO model
        results = model(image_path, show=False)  # Disable automatic display

        # Read the image
        image = cv2.imread(image_path)

        # Calculate the resize ratio based on the maximum height
        ratio = max_height / image.shape[0]  # current height of the image
        new_width = int(image.shape[1] * ratio)
        new_height = int(image.shape[0] * ratio)

        # Resize the image
        resized_image = cv2.resize(image, (new_width, new_height))

        # Process the resized image using YOLO model
        results_resized = model(resized_image, show=True)

        # Wait for a key press before moving to the next image
        cv2.waitKey(0)

# Close all OpenCV windows
cv2.destroyAllWindows()
