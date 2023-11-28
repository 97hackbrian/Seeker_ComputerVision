import cv2
import numpy as np
import urllib.request
from ultralytics import YOLO

# URL de la transmisión MJPEG (reemplaza <ip_de_raspberry> con la dirección IP de tu Raspberry Pi)
url = 'http://192.168.100.35:8080/?action=stream'

# Leer nuestro modelo
model = YOLO("/home/hackbrian/Descargas/best3.pt")

# Configurar el objeto VideoCapture para leer la transmisión MJPEG directamente desde la URL
cap = cv2.VideoCapture(url)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 15)
cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
# Bucle
while True:
    # Leer nuestros fotogramas
    ret, frame = cap.read()
    #frame=frame[:, :, 0]
    #frame = cv2.resize(frame, (200, 200))
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

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
