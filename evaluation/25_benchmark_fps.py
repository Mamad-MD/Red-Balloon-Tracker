import time
import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = r"C:\BalloonProject\runs\main\weights\best.pt"
IMG_PATH = r"C:\BalloonProject\test_big\test_scene_1.jpg"
TILE = 640
OVERLAP = 128
N_RUNS = 10

def make_tiles(W, H, tile, overlap):
    step = tile - overlap
    xs = list(range(0, max(W - tile, 0) + 1, step))
    ys = list(range(0, max(H - tile, 0) + 1, step))
    if not xs or xs[-1] + tile < W:
        xs.append(W - tile)
    if not ys or ys[-1] + tile < H:
        ys.append(H - tile)
    return [(x, y) for y in ys for x in xs]

model = YOLO(MODEL_PATH)
img = cv2.imdecode(np.fromfile(IMG_PATH, np.uint8), cv2.IMREAD_COLOR)
H, W = img.shape[:2]
tiles = make_tiles(W, H, TILE, OVERLAP)
tile_imgs = [img[y:y + TILE, x:x + TILE] for x, y in tiles]
print(f"{len(tiles)} تکه در هر فریم")

# گرم کردن GPU (این زمان حساب نمی‌شود)
_ = model.predict(tile_imgs, conf=0.25, verbose=False, device=0)
_ = model.predict(tile_imgs, conf=0.25, verbose=False, device=0)

# سنجش fp32 معمولی
times = []
for _ in range(N_RUNS):
    t0 = time.time()
    _ = model.predict(tile_imgs, conf=0.25, verbose=False, device=0)
    times.append(time.time() - t0)
avg = sum(times) / len(times)
print(f"\nfp32 معمولی: میانگین {avg*1000:.0f}ms هر فریم -> {1/avg:.2f} FPS")

# سنجش half precision (fp16)، سریع‌تر روی GPUهای جدید
times = []
for _ in range(N_RUNS):
    t0 = time.time()
    _ = model.predict(tile_imgs, conf=0.25, verbose=False, device=0, half=True)
    times.append(time.time() - t0)
avg16 = sum(times) / len(times)
print(f"fp16 (half): میانگین {avg16*1000:.0f}ms هر فریم -> {1/avg16:.2f} FPS")