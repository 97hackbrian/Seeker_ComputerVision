import torch
import torchvision.models as models
from torchvision.models.detection import yolo

# Carga tu modelo YOLO preentrenado
model = models.detection.yolo_v3(pretrained=False)
model.load_state_dict(torch.load('yolo_pretrained_weights.pth'))
model.eval()

# Ejemplo de entrada (ajusta según las necesidades de tu modelo)
dummy_input = torch.randn(1, 3, 416, 416)

# Traza el modelo con JIT
traced_model = torch.jit.trace(model, dummy_input)

# Guarda el modelo trazado
traced_model.save('yolo_traced_model.pt')
