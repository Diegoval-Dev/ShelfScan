from ultralytics import YOLO

model = YOLO("models/shelfscan_v1/weights/nvidia_best.pt")
model.train(
    data="data/dataset.yaml",
    epochs=15,
    imgsz=416,
    batch=8,
    device="cpu",
    amp=False,
    cls=2.5,
    cache="ram",
    workers=0,
    patience=8,
    name="shelfscan_v2_mps",
)
