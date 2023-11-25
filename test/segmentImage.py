import torch
import torchvision.transforms as transforms
import cv2

# 1. Cargar el modelo en PyTorch
model = torch.load('/home/hackbrian/Documentos/gitProyects/Seeker_ComputerVision/YoloTrain/runs/segment/train/weights/best.pt')  # Ajusta la ruta según donde hayas guardado tu modelo
model.eval()

# 2. Preprocesar la entrada
image = cv2.imread('/home/hackbrian/Descargas/data/dataV1/images/ground/057.jpg')  # Ajusta la ruta según tu imagen
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
transform = transforms.Compose([
    transforms.ToTensor(),
    # Añade más transformaciones si es necesario (normalización, redimensionamiento, etc.)
])
input_image = transform(image).unsqueeze(0)

# 3. Realizar la inferencia
with torch.no_grad():
    output = model(input_image)

# 4. Postprocesamiento de resultados
# (Dependerá del formato de salida específico de tu modelo YOLO)

# 5. Visualización de resultados
# (Dibuja cuadros delimitadores alrededor de los objetos detectados)

# Muestra la imagen con las detecciones
cv2.imshow('Detecciones YOLO', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
