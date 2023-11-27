/bin/.extreme_cooling start
yolo task=segment mode=train epochs=500 data=dataset.yaml model=yolov8n-seg.pt imgsz=640 batch=9
/bin/.extreme_cooling stop
