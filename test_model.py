from ultralytics import YOLO

model = YOLO("best.pt")

metrics = model.val()
print(metrics.box.map)

