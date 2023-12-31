import cv2
import sys
import time  # Importa el módulo time

sys.path.append('../Seeker_ComputerVision')

from scr.libs.YOLOSeg import YOLOSeg


# Initialize the webcam
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 200)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 200)
cap.set(cv2.CAP_PROP_FPS, 59)
cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
# Initialize YOLOv5 Instance Segmentator
model_path = "../Seeker_ComputerVision/YoloTrain/runs/segment/train/weights/best2.onnx"
yoloseg = YOLOSeg(model_path, conf_thres=0.89, iou_thres=0.8)

cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)

# Inicializa el tiempo al comienzo
start_time = time.time()

while cap.isOpened():
    # Read frame from the video
    ret, frame = cap.read()
    frame=frame[:, :, 0]
    frame = cv2.resize(frame, (200, 200))
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    
    if not ret:
        break
    

    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

    #frame = cv2.resize(frame, (640, 640))
    #input_tensor = torch.from_numpy(gray_frame).permute(2, 0, 1).float() / 255.0
    #input_tensor = input_tensor.unsqueeze(0)
    #
    # Update object localizer
    boxes, scores, class_ids, masks = yoloseg(frame)

    combined_img = yoloseg.draw_masks(frame)
    cv2.imshow("Detected Objects", combined_img)

    # Calcula los FPS
    elapsed_time = time.time() - start_time
    fps = 1 / elapsed_time
    print(f"FPS: {fps:.2f}")

    # Actualiza el tiempo para el siguiente ciclo
    start_time = time.time()

    # Press key q to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera la captura de video y cierra las ventanas
cap.release()
cv2.destroyAllWindows()
