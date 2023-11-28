import cv2
import numpy as np
import urllib.request
from ultralytics import YOLO

def apply_clahe_gaussian(image):
    # Separar los canales de la imagen en color
    b, g, r = cv2.split(image)

    # Aplicar el ecualizador CLAHE a cada canal
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(5, 5))
    b_clahe = clahe.apply(b)
    g_clahe = clahe.apply(g)
    r_clahe = clahe.apply(r)

    # Fusionar los canales de nuevo
    clahe_result = cv2.merge([b_clahe, g_clahe, r_clahe])

    # Aplicar un desenfoque gaussiano
    blurred = cv2.GaussianBlur(clahe_result, (13, 13), 0)##5,5

    return blurred


# URL de la transmisión MJPEG (reemplaza <ip_de_raspberry> con la dirección IP de tu Raspberry Pi)
url = 'http://192.168.100.35:8080/?action=stream'

# Leer nuestro modelo
model = YOLO("/home/hackbrian/gitProyects/Seeker_ComputerVision/YoloTrainV3/runs/segment/train/weights/best.pt")

# Configurar el objeto VideoCapture para leer la transmisión MJPEG directamente desde la URL
cap = cv2.VideoCapture(url)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 15)
cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
# Bucle
while True:
    # Leer nuestros fotogramas
    ret, frame = cap.read()
    #frame=frame[:, :, 0]
    #frame = cv2.resize(frame, (200, 200))
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    #frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    frame=apply_clahe_gaussian(frame)
    # Leemos resultados
    resultados = model.predict(frame, imgsz=640, conf=0.91)

    # Mostramos resultados
    anotaciones = resultados[0].plot()

    # Mostramos nuestros fotogramas
    cv2.imshow("DETECCION Y SEGMENTACION", anotaciones)

    # Cerrar nuestro programa
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
