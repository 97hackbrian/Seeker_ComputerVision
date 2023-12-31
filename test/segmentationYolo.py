# Importamos las librerias
from ultralytics import YOLO
import cv2

# Leer nuestro modelo
model = YOLO("../Seeker_ComputerVision/YoloTrain/runs/segment/train/weights/best3.pt")

# Realizar VideoCaptura
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
cap.set(cv2.CAP_PROP_FPS, 10)
cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
# Bucle
while True:
    # Leer nuestros fotogramas
    ret, frame = cap.read()
    frame=frame[:, :, 0]
    frame = cv2.resize(frame, (200, 200))
    frame = cv2.rotate(frame, cv2.ROTATE_180)

    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

    # Leemos resultados
    resultados = model.predict(frame, imgsz = 640, conf = 0.9)

    # Mostramos resultados
    anotaciones = resultados[0].plot()

    # Mostramos nuestros fotogramas
    cv2.imshow("DETECCION Y SEGMENTACION", anotaciones)

    # Cerrar nuestro programa
    if cv2.waitKey(1) == 27:
        break

        

cap.release()
cv2.destroyAllWindows()