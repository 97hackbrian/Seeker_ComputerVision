import cv2
import sys
sys.path.append('../Seeker_ComputerVision')  # Reemplaza '/ruta/al/proyecto' con la ruta real a tu proyecto

from scr.libs.YOLOSeg import YOLOSeg

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize YOLOv5 Instance Segmentator
model_path = "../Seeker_ComputerVision/YoloTrain/runs/segment/train/weights/best.onnx"
yoloseg = YOLOSeg(model_path, conf_thres=0.3, iou_thres=0.3)

cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
while cap.isOpened():

    # Read frame from the video
    ret, frame = cap.read()

    if not ret:
        break

    # Update object localizer
    boxes, scores, class_ids, masks = yoloseg(frame)

    combined_img = yoloseg.draw_masks(frame)
    cv2.imshow("Detected Objects", combined_img)

    # Press key q to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
