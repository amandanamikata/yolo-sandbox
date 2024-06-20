from roboflow import Roboflow
from ultralytics import YOLO
import torch
import gi
#gi.require_version("Gtk", "4.0")
from gi.repository import Gtk

#//DETECTION

from roboflow import Roboflow
rf = Roboflow(api_key="iCOoPmE9BfuEK86Wmqxi")
project = rf.workspace("amandanamikata").project("detection_pieces_machine_and_internet_images")
version = project.version(10)
dataset = version.download("yolov8")

v = '8n'

device = torch.device("cuda")
model = YOLO(f'yolov{v}.pt')


model.to(device)

results = model.train(
    data=r'/home/mandanamikata/Desktop/BKP_AMANDA/Yolo/detection_pieces_machine_and_internet_images-10/data.yaml',
    imgsz=640,
    epochs=20,
    batch=16,
    augment=False,
    int8=True,
    half=False,
    simplify=True,        
    name=f'yolov{v}_pieces_detect_hue_color_pieces'
)


# model = YOLO('yolov8n.pt')

# model.predict(
#    source='https://media.roboflow.com/notebooks/examples/dog.jpeg',
#    conf=0.25
# )

