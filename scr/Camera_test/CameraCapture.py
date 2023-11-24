import cv2

def apply_clahe_gaussian(gray):
    # Aplicar el ecualizador CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5, 5))
    clahe_result = clahe.apply(gray)
    # Aplicar un desenfoque gaussiano
    blurred = cv2.GaussianBlur(clahe_result, (5, 5), 0)

    return blurred

def main():
    # Configuración de la cámara con el backend V4L2
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)  # Puedes cambiar el número de la cámara según tu configuración (0 por la cámara predeterminada)
    # Configuración de parámetros
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1279)    # Ancho de la resolución 1280
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 719)   # Alto de la resolución   720
    cap.set(cv2.CAP_PROP_FPS, 59)             # FPS (cuadros por segundo)
    # Configurar la cámara para capturar en blanco y negro
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)

    if not cap.isOpened():
        print("Error al abrir la cámara.")
        return

    while True:
        ret, frame = cap.read()
        frame=frame[:, :, 0]
        frame = cv2.rotate(frame, cv2.ROTATE_180)


        if not ret:
            print("Error al capturar el fotograma.")
            break

        # Aplicar la transformación directamente a la imagen en blanco y negro
        transformed_frame = apply_clahe_gaussian(frame)

        # Mostrar el fotograma transformado en una ventana
        cv2.imshow('Transformacion', transformed_frame)

        # Salir del bucle si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar la cámara y cerrar la ventana
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
