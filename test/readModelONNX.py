import cv2
import sys
sys.path.append('../Seeker_ComputerVision')  # Reemplaza '/ruta/al/proyecto' con la ruta real a tu proyecto

from scr.libs.YOLOSeg import YOLOSeg


# Initialize YOLOv5 Instance Segmentator
model_path = "../Seeker_ComputerVision/YoloTrain/runs/segment/train/weights/best2.onnx"
yoloseg = YOLOSeg(model_path, conf_thres=0.5, iou_thres=0.3)

# Read image
img = cv2.imread("/home/hackbrian/Documentos/YoloTrain/train/images/012.jpg_ground.jpg")

# Detect Objects
boxes, scores, class_ids, masks = yoloseg(img)

# Draw detections
combined_img = yoloseg.draw_masks(img)
cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
cv2.imshow("Detected Objects", combined_img)
cv2.imwrite("doc/img/detected_objects.jpg", combined_img)
cv2.waitKey(0)
