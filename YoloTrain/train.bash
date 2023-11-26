/bin/.extreme_cooling start
yolo task=segment mode=train epochs=66 data=dataset.yaml model=yolov8s-seg.pt imgsz=640 batch=10
/bin/.extreme_cooling stop
