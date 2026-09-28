from ultralytics import YOLO

model = YOLO("YOLO26n_buoy_detector.pt")

metrics = model.val()
print(metrics.box.map)

