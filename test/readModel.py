
import onnxruntime as ort
from PIL import Image, ImageDraw
import numpy as np
import cv2


# Preparar la imagen de entrada
img = Image.open("/home/hackbrian/Documentos/YoloTrain/train/images/001.jpg_ground.jpg")
img_width, img_height = img.size
img = img.convert("RGB")
img = img.resize((640, 640))
input_data = np.array(img)
input_data = input_data.transpose(2, 0, 1)
input_data = input_data.reshape(1, 3, 640, 640).astype("float32")
input_data = input_data / 255.0

# Ejecutar el modelo
model = ort.InferenceSession("/home/hackbrian/Documentos/gitProyects/Seeker_ComputerVision/YoloTrain/runs/segment/train/weights/best.onnx")
outputs = model.run(None, {"images": input_data})

# Procesar las salidas
output0 = outputs[0][0].transpose()


boxes = output0[:, 0:84]
masks = output0[:, 84:]


masks = masks 

boxes = np.hstack([boxes, masks])

# Parsear las cajas para obtener la lista de cajas delimitadoras de objetos detectados
# Sus máscaras de segmentación y polígonos delimitadores
yolo_classes = [
    "cubo"
]

def intersection(box1, box2):
    box1_x1, box1_y1, box1_x2, box1_y2 = box1[:4]
    box2_x1, box2_y1, box2_x2, box2_y2 = box2[:4]
    x1 = max(box1_x1, box2_x1)
    y1 = max(box1_y1, box2_y1)
    x2 = min(box1_x2, box2_x2)
    y2 = min(box1_y2, box2_y2)
    return (x2 - x1) * (y2 - y1)

def union(box1, box2):
    box1_x1, box1_y1, box1_x2, box1_y2 = box1[:4]
    box2_x1, box2_y1, box2_x2, box2_y2 = box2[:4]
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    return box1_area + box2_area - intersection(box1, box2)

def iou(box1, box2):
    return intersection(box1, box2) / union(box1, box2)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def get_mask(row, box, img_width, img_height):
    mask = row.reshape(160, 160)
    mask = sigmoid(mask)
    mask = (mask > 0.5).astype("uint8") * 255

    x1, y1, x2, y2 = box
    mask_x1 = round(x1 / img_width * 160)
    mask_y1 = round(y1 / img_height * 160)
    mask_x2 = round(x2 / img_width * 160)
    mask_y2 = round(y2 / img_height * 160)

    mask = mask[mask_y1:mask_y2, mask_x1:mask_x2]
    img_mask = Image.fromarray(mask, "L")
    img_mask = img_mask.resize((round(x2 - x1), round(y2 - y1)))
    mask = np.array(img_mask)

    return mask

def get_polygon(mask):
    contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    polygon = [[contour[0][0], contour[0][1]] for contour in contours[0][0]]
    return polygon

objects = []

for row in boxes:
    xc, yc, w, h = row[:4]
    x1 = (xc - w / 2) / 640 * img_width
    y1 = (yc - h / 2) / 640 * img_height
    x2 = (xc + w / 2) / 640 * img_width
    y2 = (yc + h / 2) / 640 * img_height
    prob = row[4:84].max()

    if prob < 0.5:
        continue

    class_id = row[4:84].argmax()
    label = yolo_classes[class_id]
    mask = get_mask(row[84:25684], (x1, y1, x2, y2), img_width, img_height)
    polygon = get_polygon(mask)

    objects.append([x1, y1, x2, y2, label, prob, mask, polygon])

objects.sort(key=lambda x: x[5], reverse=True)
result = []

while len(objects) > 0:
    result.append(objects[0])
    objects = [obj for obj in objects if iou(obj, objects[0]) < 0.7]

# Visualizar los polígonos delimitadores de objetos detectados en la imagen
img = Image.open("/home/hackbrian/Documentos/YoloTrain/train/images/001.jpg_ground.jpg")
draw = ImageDraw.Draw(img, "RGBA")

for obj in result:
    [x1, y1, x2, y2, label, prob, mask, polygon] = obj

    # Mover el polígono desde (0,0) hasta el punto superior izquierdo del objeto detectado
    polygon = [(round(x1 + point[0]), round(y1 + point[1])) for point in polygon]
    
    # Dibujar el polígono en la imagen
    draw.polygon(polygon, fill=(0, 255, 0, 125))

# Mostrar la imagen con los polígonos delimitadores
img.show()

