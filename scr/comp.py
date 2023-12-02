import torch
from models.yolo import Model
from utils.general import export_tfweights

# Cargar el modelo de YOLO en formato Darknet
model = Model('path/to/your/yolov5s.yaml')  # Reemplaza con la ruta correcta a tu archivo de configuración YAML

# Exportar los pesos a un formato compatible con PyTorch
export_tfweights(model, 'path/to/your/darknet/weights')  # Reemplaza con la ruta correcta a tus pesos de Darknet
