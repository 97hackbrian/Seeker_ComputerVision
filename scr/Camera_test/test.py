import cv2

def main():
    # Configuración de la cámara con el backend V4L2
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)  # Puedes cambiar el número de la cámara según tu configuración (0 por la cámara predeterminada)

    # Configuración de parámetros
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)    # Ancho de la resolución
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)   # Alto de la resolución
    cap.set(cv2.CAP_PROP_FPS, 50)             # FPS (cuadros por segundo)

    if not cap.isOpened():
        print("Error al abrir la cámara.")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error al capturar el fotograma.")
            break

        # Aquí puedes realizar operaciones en el fotograma si es necesario
        # Por ejemplo, mostrar el fotograma en una ventana
        cv2.imshow('Camara', frame)

        # Salir del bucle si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar la cámara y cerrar la ventana
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
