from roboflow import Roboflow
from ultralytics import YOLO
import torch


rf = Roboflow(api_key="iCOoPmE9BfuEK86Wmqxi")
project = rf.workspace("amandanamikata").project("full-lengh-detect")
version = project.version(2)
dataset = version.download("yolov8")


v = '8n'

device = torch.device("cuda")
model = YOLO(f'yolov{v}.pt')


model.to(device)

results = model.train(
    data=r'/home/amandamnamikata/Yolo/full-lengh-detect-2/data.yaml',
    imgsz=640,
    epochs=50,
    batch=16,
    name=f'yolov{v}_full_lenght'
)


# model = YOLO('yolov8n.pt')

# model.predict(
#    source='https://media.roboflow.com/notebooks/examples/dog.jpeg',
#    conf=0.25
# )

