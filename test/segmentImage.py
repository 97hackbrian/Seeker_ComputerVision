import torch
import cv2
# Loading in yolov5s - you can switch to larger models such as yolov5m or yolov5l, or smaller such as yolov5n
model = torch.hub.load('ultralytics/yolov8n-seg.pt', 'custom', path='/home/hackbrian/Documentos/gitProyects/Seeker_ComputerVision/YoloTrain/runs/segment/train/weights/best.pt')

img = '/home/hackbrian/Documentos/YoloTrain/train/images/001.jpg_ground.jpg'  # or file, Path, PIL, OpenCV, numpy, list
results = model(img)
#results = model.predict(img, imgsz = 640, conf = 0.83)
anotaciones = results[0].plot()

# Mostramos nuestros fotogramas
cv2.imshow("DETECCION Y SEGMENTACION", anotaciones)
