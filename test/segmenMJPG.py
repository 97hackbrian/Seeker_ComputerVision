import cv2
import numpy as np
import urllib.request
from ultralytics import YOLO

# URL de la transmisión MJPEG (reemplaza <ip_de_raspberry> con la dirección IP de tu Raspberry Pi)
url = 'http://<ip_de_raspberry>:8080/?action=stream'

# Leer nuestro modelo
model = YOLO("/home/hackbrian/Documentos/train2/runs/segment/train/weights/best.pt")

# Configurar el objeto VideoCapture para leer la transmisión MJPEG directamente desde la URL
cap = cv2.VideoCapture(url)

# Bucle
while True:
    # Leer nuestros fotogramas
    ret, frame = cap.read()

    # Leemos resultados
    resultados = model.predict(frame, imgsz=640, conf=0.8)

    # Mostramos resultados
    anotaciones = resultados[0].plot()

    # Mostramos nuestros fotogramas
    cv2.imshow("DETECCION Y SEGMENTACION", anotaciones)

    # Cerrar nuestro programa
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
