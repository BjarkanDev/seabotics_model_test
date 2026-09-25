from ultralytics import YOLO

model = YOLO("yolo26n.pt")

results = model.train(
        data="dataset/dataset.yaml",
        epochs=100,
        imgsz=[352,640],
        device="cpu"
        )
