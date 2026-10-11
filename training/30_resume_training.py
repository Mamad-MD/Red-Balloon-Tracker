from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"C:\BalloonProject\runs\main\weights\last.pt")
    model.train(resume=True)