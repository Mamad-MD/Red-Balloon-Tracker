from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo26s-p2.yaml").load("yolo26s.pt")
    model.train(
        data=r"C:\BalloonProject\dataset_final\data.yaml",
        imgsz=640,
        epochs=3,
        batch=16,
        workers=4,
        device=0,
        project=r"C:\BalloonProject\runs",
        name="smoke",
        exist_ok=True,
    )