import cv2
import urllib.request
import numpy as np

# URL de la transmisión MJPEG (sustituye <ip_de_raspberry> con la dirección IP de tu Raspberry Pi)
url = 'http://192.168.100.35:8080/?action=stream'

# Configura el objeto VideoCapture para leer la transmisión MJPEG directamente desde la URL
cap = cv2.VideoCapture(url)

while True:
    # Lee un frame de la transmisión
    ret, frame = cap.read()

    # Muestra el frame
    cv2.imshow('Video desde Raspberry Pi', frame)

    # Si se presiona la tecla 'q', sale del bucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera el objeto VideoCapture y cierra la ventana
cap.release()
cv2.destroyAllWindows()
