from roboflow import Roboflow
from ultralytics import YOLO
import torch
import gi
#gi.require_version("Gtk", "4.0")
from gi.repository import Gtk

#//SEGMENTATION

rf = Roboflow(api_key="iCOoPmE9BfuEK86Wmqxi")
project = rf.workspace("amandanamikata").project("outline_volume_segmentation")
version = project.version(5)
dataset = version.download("yolov8")


v = '8n'

device = torch.device("cuda")
model = YOLO(f'yolov{v}-seg.pt')


model.to(device)

results = model.train(
    data=r'/home/mandanamikata/Desktop/BKP_AMANDA/Yolo/outline_volume_segmentation-5/data.yaml',
    imgsz=640,
    epochs=25,
    batch=16,
    augment=False,
    int8=True,
    half=False,
    simplify=True, 
    name=f'yolov{v}_seg_vol_custom'
)






# model = YOLO('yolov8n.pt')

# model.predict(
#    source='https://media.roboflow.com/notebooks/examples/dog.jpeg',
#    conf=0.25
# )

