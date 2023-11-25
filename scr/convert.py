import torch

# Cargar el modelo YOLO preentrenado
model = torch.load('/home/hackbrian/Documentos/gitProyects/Seeker_ComputerVision/YoloTrain/runs/segment/train/weights/best.pt')

# Convertir el modelo a formato ONNX
dummy_input = torch.randn(1, 3, 640, 640)  # Puedes ajustar las dimensiones según tus datos de entrada
torch.onnx.export(model, dummy_input, '/home/hackbrian/Documentos/gitProyects/Seeker_ComputerVision/YoloTrain/runs/segment/train/weights/best.onnx', opset_version=11)
