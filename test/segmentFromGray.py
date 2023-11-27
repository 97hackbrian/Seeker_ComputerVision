# Importamos las librerias
from ultralytics import YOLO
import cv2
import torch
import numpy as np

def apply_clahe_gaussian(gray):
    # Aplicar el ecualizador CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5, 5))
    clahe_result = clahe.apply(gray)
    # Aplicar un desenfoque gaussiano
    blurred = cv2.GaussianBlur(clahe_result, (5, 5), 0)
    #return clahe_result
    return blurred

# Leer nuestro modelo
model = YOLO("/home/hackbrian/gitProyects/Seeker_ComputerVision/YoloTrainV3/runs/segment/train/weights/best.pt")

# Realizar VideoCaptura
cap = cv2.VideoCapture(0)

# Bucle
while True:
    # Leer nuestros fotogramas
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #gray_frame=apply_clahe_gaussian(gray_frame)

    # Aplicar preprocesamiento a la imagen en escala de grises si es necesario
    # gray_frame = apply_clahe_gaussian(gray_frame)

    # Expandir dimensiones para que tenga tres canales (shape: HxW -> HxWx3)
    gray_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    # Cambiar el tamaño de la imagen a (640, 640)
    #gray_frame = cv2.resize(gray_frame, (640, 640))

    # Convertir a tensor y normalizar la imagen (si es necesario)
    input_tensor = torch.from_numpy(gray_frame).permute(2, 0, 1).float() / 255.0

    # Agregar una dimensión para el lote (shape: 3x640x640 -> 1x3x640x640)
    input_tensor = input_tensor.unsqueeze(0)

    # Leemos resultados
    resultados = model.predict(input_tensor, imgsz=640, conf=0.88)

    # Mostramos resultados
    anotaciones = resultados[0].plot()

    # Mostramos nuestros fotogramas
    cv2.imshow("DETECCION Y SEGMENTACION", anotaciones)

    # Cerrar nuestro programa
    if cv2.waitKey(1) == 27:
        break