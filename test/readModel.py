import cv2
import numpy as np
import onnxruntime
from PIL import Image, ImageDraw

# Ruta al archivo ONNX
onnx_model_path = '/home/hackbrian/gitProyects/Seeker_ComputerVision/YoloTrain/runs/segment/train/weights/best.onnx'

# Carga del modelo ONNX
session = onnxruntime.InferenceSession(onnx_model_path)

# Captura de video desde la cámara
cap = cv2.VideoCapture(0)  # El argumento 0 indica la cámara predeterminada

# Tamaño deseado del tensor de entrada
target_size = (640, 640)

def intersection(box1,box2):
    box1_x1,box1_y1,box1_x2,box1_y2 = box1[:4]
    box2_x1,box2_y1,box2_x2,box2_y2 = box2[:4]
    x1 = max(box1_x1,box2_x1)
    y1 = max(box1_y1,box2_y1)
    x2 = min(box1_x2,box2_x2)
    y2 = min(box1_y2,box2_y2)
    return (x2-x1)*(y2-y1) 

def union(box1,box2):
    box1_x1,box1_y1,box1_x2,box1_y2 = box1[:4]
    box2_x1,box2_y1,box2_x2,box2_y2 = box2[:4]
    box1_area = (box1_x2-box1_x1)*(box1_y2-box1_y1)
    box2_area = (box2_x2-box2_x1)*(box2_y2-box2_y1)
    return box1_area + box2_area - intersection(box1,box2)

def iou(box1,box2):
    return intersection(box1,box2)/union(box1,box2)

def sigmoid(z):
    return 1/(1 + np.exp(-z))

# parse segmentation mask
def get_mask(row, box, img_width, img_height):
    # convert mask to image (matrix of pixels)
    mask = row.reshape(160,160)
    mask = sigmoid(mask)
    mask = (mask > 0.5).astype("uint8")*255
    # crop the object defined by "box" from mask
    x1,y1,x2,y2 = box
    mask_x1 = round(x1/img_width*160)
    mask_y1 = round(y1/img_height*160)
    mask_x2 = round(x2/img_width*160)
    mask_y2 = round(y2/img_height*160)
    mask = mask[mask_y1:mask_y2,mask_x1:mask_x2]
    # resize the cropped mask to the size of object
    img_mask = Image.fromarray(mask,"L")
    img_mask = img_mask.resize((round(x2-x1),round(y2-y1)))
    mask = np.array(img_mask)
    return mask

# calculate bounding polygon from mask
def get_polygon(mask):
    contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    polygon = [[contour[0][0],contour[0][1]] for contour in contours[0][0]]
    return polygon

# parse and filter all boxes


while True:
    # Captura de un fotograma
    ret, frame = cap.read()

    # Preprocesamiento del fotograma según las necesidades del modelo
    # Cambia el tamaño de la imagen
    resized_frame = cv2.resize(frame, target_size)
    
    # Normalización y cambio de formato
    input_data = resized_frame.astype(np.float32) / 255.0
    input_data = np.transpose(input_data, (2, 0, 1))  # Cambia el orden de las dimensiones
    input_data = np.expand_dims(input_data, axis=0)

    # Realiza la inferencia en el fotograma preprocesado
    output = session.run(None, {'images': input_data})

    output0 = output[0]
    output1 = output[1]
    output0 = output0[0].transpose()
    output1 = output1[0]
    boxes = output0[:, 0:84]
    masks = output0[:, 84:]
    output1 = output1.reshape(32, 160 * 160)
    output1_transposed = output1.T

    # Multiplicación de matrices
    masks = masks * output1_transposed

    boxes = np.hstack([boxes, masks])

    yolo_classes = ["cubo"]
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

    # apply non-maximum suppression
    objects.sort(key=lambda x: x[5], reverse=True)
    result = []

    while len(objects) > 0:
        result.append(objects[0])
        objects = [obj for obj in objects if iou(obj, objects[0]) < 0.7]

    for obj in result:
        [x1, y1, x2, y2, label, prob, mask, polygon] = obj
        polygon = [(round(x1 + point[0]), round(y1 + point[1])) for point in polygon]
        frame = cv2.polylines(frame, [np.array(polygon)], isClosed=True, color=(0, 255, 0), thickness=2)

    cv2.imshow('Segmentación de Objetos', frame)

    # Detener el bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera los recursos
cap.release()
cv2.destroyAllWindows()