# Importamos las librerias
from ultralytics import YOLO
import cv2

def apply_clahe_gaussian(gray):
    # Aplicar el ecualizador CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5, 5))
    clahe_result = clahe.apply(gray)
    # Aplicar un desenfoque gaussiano
    blurred = cv2.GaussianBlur(clahe_result, (5, 5), 0)

    return blurred

# Leer nuestro modelo
model = YOLO("/home/hackbrian/Documentos/train2/runs/segment/train/weights/best.pt")#2

# Realizar VideoCaptura
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1279)    # Ancho de la resolución 1280
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 719)   # Alto de la resolución   720
cap.set(cv2.CAP_PROP_FPS, 59)             # FPS (cuadros por segundo)
# Configurar la cámara para capturar en blanco y negro
cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)

# Bucle
while True:
    # Leer nuestros fotogramas
    ret, frame = cap.read()
    frame=frame[:, :, 0]
    frame = apply_clahe_gaussian(frame)

    # Leemos resultados
    resultados = model.predict(frame, imgsz = 640, conf = 0.8)

    # Mostramos resultados
    anotaciones = resultados[0].plot()

    # Mostramos nuestros fotogramas
    cv2.imshow("DETECCION Y SEGMENTACION", anotaciones)

    # Cerrar nuestro programa
    if cv2.waitKey(1) == 27:
        break

        

cap.release()
cv2.destroyAllWindows()