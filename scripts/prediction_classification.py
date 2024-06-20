from ultralytics import YOLO



model = YOLO(r'/home/amandamnamikata/Yolo/runs/classify/train_dark_imgs/weights/best.pt')

img = r'/home/amandamnamikata/Yolo/dataset/teste_classify/alco1.jpg'
img1 = r'/home/amandamnamikata/Yolo/dataset/teste_classify/alco2.jpg'
img2 = r'/home/amandamnamikata/Yolo/dataset/teste_classify/clean1.jpg'
img3 = r'/home/amandamnamikata/Yolo/dataset/teste_classify/clean2.jpg'
img4 = r'/home/amandamnamikata/Yolo/dataset/teste_classify/clean3.jpg'
img5 = r'/home/amandamnamikata/Yolo/dataset/teste_classify/cup1.jpg'
img6 = r'/home/amandamnamikata/Yolo/dataset/teste_classify/cup2.jpg'
img7 = r'/home/amandamnamikata/Yolo/dataset/teste_classify/latinha1.jpg'
img8 = r'/home/amandamnamikata/Yolo/dataset/teste_classify/latinha2.jpg'
img9 = r'/home/amandamnamikata/Yolo/dataset/teste_classify/refri1.jpg'
img10 = r'/home/amandamnamikata/Yolo/dataset/teste_classify/refri2.jpg'
img11 = r'/home/amandamnamikata/Yolo/dataset/teste_classify/refri3.jpg'
img12 = r'/home/amandamnamikata/Yolo/dataset/teste_classify/refri4.jpg'
img13 = r'/home/amandamnamikata/Yolo/dataset/teste_classify/vidro.jpg'
img14 = r'/home/amandamnamikata/Yolo/dataset/teste_classify/vidro1.jpg'
img15 = r'/home/amandamnamikata/Yolo/dataset/teste_classify/termica.jpg'
img16 = r'/home/amandamnamikata/Yolo/dataset/teste_classify/airfrier.jpg'
img17 = r'/home/amandamnamikata/Yolo/dataset/teste_classify/pote.jpg'

print("-------- 50 SHADES DARKER ------------ ")

results = model.predict(source=img)
results = model.predict(source=img1)
results = model.predict(source=img2)
results = model.predict(source=img3)
results = model.predict(source=img4)
results = model.predict(source=img5)
results = model.predict(source=img6)
results = model.predict(source=img7)
results = model.predict(source=img8)
results = model.predict(source=img9)
results = model.predict(source=img10)
results = model.predict(source=img11)
results = model.predict(source=img12)
results = model.predict(source=img13)
results = model.predict(source=img14)
results = model.predict(source=img15)
results = model.predict(source=img16)
results = model.predict(source=img17)

print("---------------------------")
     
model = YOLO(r'/home/amandamnamikata/Yolo/runs/classify/train_grayscale_imgs_1/weights/best.pt')

print("-------- GREYSCALE ------------ ")

results = model.predict(source=img)
results = model.predict(source=img1)
results = model.predict(source=img2)
results = model.predict(source=img3)
results = model.predict(source=img4)
results = model.predict(source=img5)
results = model.predict(source=img6)
results = model.predict(source=img7)
results = model.predict(source=img8)
results = model.predict(source=img9)
results = model.predict(source=img10)
results = model.predict(source=img11)
results = model.predict(source=img12)
results = model.predict(source=img13)
results = model.predict(source=img14)
results = model.predict(source=img15)
results = model.predict(source=img16)
results = model.predict(source=img17)

print("---------------------------")

model = YOLO(r'/home/amandamnamikata/Yolo/runs/classify/train_cut_imgs_3/weights/best.pt')

print("-------- CUT IMAGES ------------ ")

results = model.predict(source=img)
results = model.predict(source=img1)
results = model.predict(source=img2)
results = model.predict(source=img3)
results = model.predict(source=img4)
results = model.predict(source=img5)
results = model.predict(source=img6)
results = model.predict(source=img7)
results = model.predict(source=img8)
results = model.predict(source=img9)
results = model.predict(source=img10)
results = model.predict(source=img11)
results = model.predict(source=img12)
results = model.predict(source=img13)
results = model.predict(source=img14)
results = model.predict(source=img15)
results = model.predict(source=img16)
results = model.predict(source=img17)

print("---------------------------")