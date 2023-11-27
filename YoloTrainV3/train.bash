/bin/.extreme_cooling start
yolo task=segment mode=train epochs=1000 data=dataset.yaml model=yolov8n-seg.pt imgsz=640 batch=3
/bin/.extreme_cooling stop
