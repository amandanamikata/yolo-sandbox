import os
import cv2

def crop_image_yolo(image_path, coordinates, output_path):
    # Read image
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    # Read YOLO coordinates
    class_id, x_center, y_center, box_width, box_height = map(float, coordinates.split())

    # Convert YOLO coordinates to pixel coordinates
    x1 = int((x_center - box_width / 2) * width)
    y1 = int((y_center - box_height / 2) * height)
    x2 = int((x_center + box_width / 2) * width)
    y2 = int((y_center + box_height / 2) * height)

    # Crop the image
    cropped_image = image[y1:y2, x1:x2]

    # Resize the cropped region to 224x224
    resized_image = cv2.resize(cropped_image, (224, 224))

    # Save the cropped and resized image
    filename = os.path.basename(image_path)
    output_filename = os.path.join(output_path, filename)
    cv2.imwrite(output_filename, resized_image)

# Paths
image_dir = 'C:\\Users\\amanda\\Desktop\\bottles\\segmented\\images'
coordinates_dir = 'C:\\Users\\amanda\\Desktop\\bottles\\segmented\\labels'
output_dir = 'C:\\Users\\amanda\\Desktop\\bottles\\script\\gargalos'

# Read YOLO coordinates files in the directory
for filename in os.listdir(coordinates_dir):
    if filename.endswith('.txt'):
        coordinates_path = os.path.join(coordinates_dir, filename)
        print("Coordinates file:", coordinates_path)
        
        # Read YOLO coordinates file
        with open(coordinates_path, 'r') as f:
            coordinates_lines = f.readlines()

        # Filter coordinates for the "boca" class
        boca_coordinates = [line.strip() for line in coordinates_lines if line.startswith('0')]

        # Crop images based on boca class coordinates
        for coordinates in boca_coordinates:
            image_filename = filename.replace('.txt', '.jpg')
            image_path = os.path.join(image_dir, image_filename)
            crop_image_yolo(image_path, coordinates, output_dir)
