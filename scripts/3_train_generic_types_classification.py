from roboflow import Roboflow
from ultralytics import YOLO

rf = Roboflow(api_key="iCOoPmE9BfuEK86Wmqxi")
project = rf.workspace("amandanamikata").project("classification-of-types")
version = project.version(2)
dataset = version.download("folder")

model = YOLO("yolov8n-cls.pt")

model.train(data="/home/amandamnamikata/Yolo/classification-of-types--2", epochs=45, batch=16, augment=True)