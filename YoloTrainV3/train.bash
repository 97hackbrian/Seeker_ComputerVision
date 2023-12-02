/bin/.extreme_cooling start
yolo task=segment mode=train epochs=1000 patience=300 data=dataset.yaml model=yolov8s-seg.pt imgsz=640 batch=15
/bin/.extreme_cooling stop
