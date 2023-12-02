import torch
import torchvision.transforms as transforms
import cv2
import numpy as np

# 1. Cargar el modelo preentrenado
model = torch.load('/home/hackbrian/Documentos/gitProyects/Seeker_ComputerVision/YoloTrain/runs/segment/train/weights/best.pt')  # Ajusta la ruta según donde hayas guardado tu modelo
model.eval()

# 2. Configurar la transformación de la imagen
transform = transforms.Compose([
    transforms.ToTensor(),
    # Añade más transformaciones si es necesario (normalización, redimensionamiento, etc.)
])

# 3. Inicializar la cámara
cap = cv2.VideoCapture(0)  # Utiliza 0 para la cámara predeterminada, ajusta según sea necesario

while True:
    # 4. Capturar el cuadro de la cámara
    ret, frame = cap.read()

    # 5. Preprocesar la imagen
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_image = transform(image).unsqueeze(0)

    # 6. Realizar la inferencia
    with torch.no_grad():
        output = model(input_image)

    # 7. Procesar la salida (depende de tu modelo y formato de salida)

    # 8. Visualizar la máscara de segmentación
    mask = output[0].numpy()  # Ajusta según el formato de salida de tu modelo
    mask = np.argmax(mask, axis=0)  # Puedes necesitar ajustar esta línea según el formato de salida

    # 9. Aplicar la máscara a la imagen original
    masked_frame = cv2.bitwise_and(frame, frame, mask=(mask > 0).astype(np.uint8))

    # 10. Mostrar la imagen original y la imagen segmentada
    cv2.imshow('Original', frame)
    cv2.imshow('Segmentación', masked_frame)

    # 11. Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 12. Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
