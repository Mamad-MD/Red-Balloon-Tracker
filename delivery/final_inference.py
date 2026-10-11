# -*- coding: utf-8 -*-
"""
============================================================================
 کد نهایی اجرا - سیستم تشخیص و ردیابی بالن قرمز
============================================================================
این اسکریپت روی یک ویدیوی ورودی (از دوربین ثابت) اجرا می‌شود و کارهای زیر
را به‌ترتیب انجام می‌دهد:

  1) هر فریم را به تکه‌های 640x640 با هم‌پوشانی می‌برد (چون بالن دور خیلی
     ریز است و resize کردن کل فریم آن را نابود می‌کند)
  2) با ترکیب «تشخیص حرکت» و «بررسی رنگ قرمز»، فقط تکه‌های مشکوک را
     انتخاب می‌کند (نه همه‌ی 48 تکه)، تا سرعت اجرا بالا برود
  3) فقط همان تکه‌های انتخاب‌شده را به مدل YOLO می‌دهد
  4) نتایج تکه‌های هم‌پوشان را با NMS ادغام می‌کند
  5) فاصله‌ی هر بالن را از روی قطر پیکسلی تخمین می‌زند (با ضریب اصلاح‌شده)
  6) با یک ردیاب ساده (Centroid Tracker) به هر بالن یک شناسه‌ی پایدار
     (ID) می‌دهد که در طول فریم‌ها ثابت می‌ماند
  7) ویدیوی خروجی با کادر/ID/فاصله رسم‌شده را ذخیره می‌کند، و همچنین یک
     فایل CSV از همه‌ی تشخیص‌ها برای تحلیل و لاگ بعدی

============================================================================
 نکات مهم و محدودیت‌های شناخته‌شده (لطفاً قبل از استفاده بخوانید)
============================================================================
- ضریب تصحیح فاصله فقط روی یک عکس تست مصنوعی (6 بالن، فواصل 150 تا 800
  متر) کالیبره شده است. کالیبراسیون دقیق نیاز به عکس واقعی از همین دوربین
  در فواصل مشخص دارد. خطای گزارش‌شده در بازه‌ی 150-650 متر زیر 6% است.
- سرعت (FPS) روی ویدیوهای شبیه‌سازی‌شده (نه فیلم واقعی دوربین IMX477)
  بین 7.7 تا 9.2 فریم بر ثانیه سنجیده شده؛ هدف کارفرما 12 FPS است. این
  عدد باید حتماً با فیلم واقعی دوباره سنجیده شود، چون رفتار نویز واقعی
  دوربین با نویز شبیه‌سازی‌شده می‌تواند فرق داشته باشد.
- ردیاب فعلی ساده (بر مبنای نزدیکی موقعیت) است، نه ByteTrack/BoT-SORT
  کامل؛ برای سرعت آهسته‌ی بالن در آسمان کافی است اما اگر چند بالن هم‌زمان
  از فاصله‌ی نزدیک به هم عبور کنند ممکن است ID جابه‌جا شود.
============================================================================
"""

import csv
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ============================================================
# تنظیمات قابل‌تغییر - اینجا را بر اساس نیاز خودتان عوض کنید
# ============================================================


import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = str(BASE_DIR / "best.pt")
VIDEO_PATH = str(BASE_DIR / "input.mp4")
OUT_VIDEO_PATH = str(BASE_DIR / "output" / "result_video.mp4")
OUT_CSV_PATH = str(BASE_DIR / "output" / "detections_log.csv")


# --- تکه‌بندی ---
TILE_SIZE = 640
TILE_OVERLAP = 128

# --- فیلتر حرکت + رنگ (برای انتخاب تکه‌های مشکوک) ---
BLUR_SIGMA = 3.0            # هموارسازی قبل از مقایسه‌ی فریم‌ها (نویز را کم می‌کند)
MOTION_THRESH = 45          # آستانه‌ی شدت اختلاف روشنایی برای "پیکسل متحرک"
MOTION_MIN_PIXELS = 20      # حداقل تعداد پیکسل متحرک در یک تکه تا مشکوک شمرده شود
RED_MIN_PIXELS = 5          # حداقل تعداد پیکسل قرمزِ مشکوک در یک تکه

# --- مدل ---
CONF_THRESH = 0.25
IOU_NMS = 0.4

# --- تخمین فاصله ---
FOCAL_PX = 3871             # فاصله‌ی کانونی معادل به پیکسل (دوربین IMX477 + لنز 6mm)
BALLOON_DIAM_M = 1.5        # قطر واقعی بالن (متر)
DIST_ACCURACY_PCT = 15      # عدم قطعیت تخمینی فاصله (بدون کالیبراسیون واقعی لنز)
# ضریب تصحیح خطای سیستماتیک اندازه‌ی کادر (کالیبره‌شده روی عکس تست 6 بالنه):
# قطر واقعی مدل کمی بزرگ‌تر از بالن واقعی پیش‌بینی می‌شود؛ این دو عدد آن را اصلاح می‌کنند.
DIAM_CORRECTION_SLOPE = 1.025
DIAM_CORRECTION_OFFSET = 1.41

# --- ردیاب (Tracker) ---
TRACK_MAX_MATCH_DIST = 80   # حداکثر فاصله‌ی پیکسلی برای تطبیق بالن بین دو فریم پیاپی
TRACK_MAX_MISSED_FRAMES = 5  # بعد از این تعداد فریمِ ندیدن، ID حذف می‌شود
TRACK_MIN_HITS_TO_CONFIRM = 2  # حداقل تعداد فریم دیده‌شدن تا ID تأیید و نمایش داده شود


# ============================================================
# توابع کمکی
# ============================================================

def make_tiles(w, h, tile, overlap):
    """لیست مختصات (x, y) گوشه‌ی بالا-چپ هر تکه را برمی‌گرداند."""
    step = tile - overlap
    xs = list(range(0, max(w - tile, 0) + 1, step))
    ys = list(range(0, max(h - tile, 0) + 1, step))
    if not xs or xs[-1] + tile < w:
        xs.append(w - tile)
    if not ys or ys[-1] + tile < h:
        ys.append(h - tile)
    return [(x, y) for y in ys for x in xs]


def count_red_pixels(tile_bgr):
    """تعداد پیکسل‌های "مشکوک به قرمز بالن" را در یک تکه می‌شمارد."""
    hsv = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = ((h <= 10) | (h >= 170)) & (s >= 100) & (v >= 60)
    return int(mask.sum())


def select_suspicious_tiles(frame, prev_gray_blurred, tiles):
    """
    تکه‌هایی را انتخاب می‌کند که یا حرکت کافی دارند یا رنگ قرمز مشکوک.
    خروجی: (لیست تکه‌های تصویر, لیست مختصات آن‌ها, فریم خاکستری‌بلورشده‌ی فعلی)
    """
    gray_blurred = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (0, 0), BLUR_SIGMA)
    diff = cv2.absdiff(gray_blurred, prev_gray_blurred)

    selected_tiles, selected_xy = [], []
    for x, y in tiles:
        d = diff[y:y + TILE_SIZE, x:x + TILE_SIZE]
        motion_px = int((d > MOTION_THRESH).sum())
        tile_bgr = frame[y:y + TILE_SIZE, x:x + TILE_SIZE]
        red_px = count_red_pixels(tile_bgr)
        if motion_px >= MOTION_MIN_PIXELS or red_px >= RED_MIN_PIXELS:
            selected_tiles.append(tile_bgr)
            selected_xy.append((x, y))
    return selected_tiles, selected_xy, gray_blurred


def run_model_on_tiles(model, tiles_imgs, tiles_xy):
    """مدل را روی تکه‌های انتخاب‌شده اجرا می‌کند و باکس‌ها را به مختصات کل فریم برمی‌گرداند."""
    boxes, scores = [], []
    if not tiles_imgs:
        return boxes, scores
    results = model.predict(tiles_imgs, conf=CONF_THRESH, verbose=False)
    for (x, y), res in zip(tiles_xy, results):
        for box in res.boxes:
            bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
            boxes.append([bx1 + x, by1 + y, bx2 + x, by2 + y])
            scores.append(float(box.conf[0]))
    return boxes, scores


def nms(boxes, scores, iou_thresh):
    """ادغام باکس‌های هم‌پوشان (که ممکن است در دو تکه‌ی مجاور تکراری دیده شده باشند)."""
    if not boxes:
        return []
    boxes_arr = np.array(boxes, dtype=np.float32)
    scores_arr = np.array(scores, dtype=np.float32)
    idxs = scores_arr.argsort()[::-1]
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        if len(idxs) == 1:
            break
        rest = idxs[1:]
        xx1 = np.maximum(boxes_arr[i, 0], boxes_arr[rest, 0])
        yy1 = np.maximum(boxes_arr[i, 1], boxes_arr[rest, 1])
        xx2 = np.minimum(boxes_arr[i, 2], boxes_arr[rest, 2])
        yy2 = np.minimum(boxes_arr[i, 3], boxes_arr[rest, 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area_i = (boxes_arr[i, 2] - boxes_arr[i, 0]) * (boxes_arr[i, 3] - boxes_arr[i, 1])
        area_r = (boxes_arr[rest, 2] - boxes_arr[rest, 0]) * (boxes_arr[rest, 3] - boxes_arr[rest, 1])
        iou = inter / (area_i + area_r - inter + 1e-6)
        idxs = rest[iou <= iou_thresh]
    return keep


def estimate_distance(d_px):
    """
    فاصله‌ی بالن را از روی قطر پیکسلی کادر تخمین می‌زند.
    ابتدا خطای سیستماتیک اندازه‌ی کادر را با ضریب کالیبره‌شده اصلاح می‌کند.
    """
    if d_px <= 0:
        return None
    d_corrected = (d_px - DIAM_CORRECTION_OFFSET) / DIAM_CORRECTION_SLOPE
    if d_corrected <= 0:
        return None
    return FOCAL_PX * BALLOON_DIAM_M / d_corrected


class CentroidTracker:
    """
    ردیاب ساده: هر بالن تشخیص‌داده‌شده در فریم فعلی را با نزدیک‌ترین بالن
    فریم قبلی تطبیق می‌دهد و شناسه‌ی (ID) پایدار می‌دهد. مناسب برای اشیای
    آهسته و کم‌تعداد مثل بالن در آسمان.
    """

    def __init__(self, max_dist, max_missed, min_hits):
        self.max_dist = max_dist
        self.max_missed = max_missed
        self.min_hits = min_hits
        self.next_id = 1
        self.tracks = {}  # id -> {"center","missed","hits","history","box"}

    def update(self, detections):
        """detections: لیست (cx, cy, box). خروجی: لیست (id, cx, cy, box, history)."""
        unmatched = list(range(len(detections)))
        matched_ids = set()

        for tid, track in list(self.tracks.items()):
            best_j, best_dist = None, self.max_dist
            for j in unmatched:
                cx, cy, _ = detections[j]
                tx, ty = track["center"]
                dist = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist, best_j = dist, j
            if best_j is not None:
                cx, cy, box = detections[best_j]
                track["center"] = (cx, cy)
                track["box"] = box
                track["missed"] = 0
                track["hits"] += 1
                track["history"].append((cx, cy))
                if len(track["history"]) > 30:
                    track["history"].pop(0)
                unmatched.remove(best_j)
                matched_ids.add(tid)
            else:
                track["missed"] += 1

        for tid in list(self.tracks.keys()):
            if self.tracks[tid]["missed"] > self.max_missed:
                del self.tracks[tid]

        for j in unmatched:
            cx, cy, box = detections[j]
            self.tracks[self.next_id] = {
                "center": (cx, cy), "box": box, "missed": 0,
                "hits": 1, "history": [(cx, cy)],
            }
            matched_ids.add(self.next_id)
            self.next_id += 1

        out = []
        for tid in matched_ids:
            t = self.tracks[tid]
            if t["missed"] == 0 and t["hits"] >= self.min_hits:
                out.append((tid, t["center"][0], t["center"][1], t["box"], t["history"]))
        return out


# ============================================================
# اجرای اصلی
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=VIDEO_PATH, help="Path to input video")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to YOLO model")
    parser.add_argument("--output-video", default=OUT_VIDEO_PATH, help="Path to output video")
    parser.add_argument("--output-csv", default=OUT_CSV_PATH, help="Path to output CSV")
    args = parser.parse_args()
    Path(args.output_video).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)

    print(f"در حال بارگذاری مدل از: {args.model}")
    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"نتوانستم ویدیو را باز کنم: {args.source}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 10
    tiles = make_tiles(w, h, TILE_SIZE, TILE_OVERLAP)
    print(f"ابعاد ویدیو: {w}x{h} | تعداد تکه در هر فریم: {len(tiles)} | FPS ورودی: {fps_in:.1f}")

    ok, first_frame = cap.read()
    if not ok:
        raise RuntimeError("ویدیو خالی است یا خوانده نشد.")
    prev_gray_blurred = cv2.GaussianBlur(
        cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY), (0, 0), BLUR_SIGMA)

    writer = cv2.VideoWriter(args.output_video, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"نتوانستم فایل ویدیوی خروجی را ایجاد کنم: {args.output_video}")
    tracker = CentroidTracker(TRACK_MAX_MATCH_DIST, TRACK_MAX_MISSED_FRAMES, TRACK_MIN_HITS_TO_CONFIRM)
    colors = {}

    csv_rows = []
    frame_idx = 0
    frame_times = []

    # فریم اول را هم پردازش می‌کنیم (چون قبلاً فقط برای مقداردهی اولیه خوانده شد)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        t0 = time.time()

        sel_tiles, sel_xy, gray_blurred = select_suspicious_tiles(frame, prev_gray_blurred, tiles)
        raw_boxes, raw_scores = run_model_on_tiles(model, sel_tiles, sel_xy)
        keep_idx = nms(raw_boxes, raw_scores, IOU_NMS)

        detections = []
        for i in keep_idx:
            x1, y1, x2, y2 = raw_boxes[i]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            detections.append((cx, cy, (x1, y1, x2, y2)))

        tracked = tracker.update(detections)

        vis = frame.copy()
        for tid, cx, cy, (x1, y1, x2, y2), history in tracked:
            d_px = max(x2 - x1, y2 - y1)
            dist_m = estimate_distance(d_px)


                        # ===== نمایش تمیز و مینیمال بالن =====
            TRACK_COLORS = [
                (255, 255, 0),   # Cyan
                (0, 255, 255),   # Yellow
                (255, 0, 255),   # Magenta
                (0, 255, 0)      # Green
            ]

            if tid not in colors:
                colors[tid] = TRACK_COLORS[(tid - 1) % len(TRACK_COLORS)]
            color = colors[tid]

            ix1, iy1, ix2, iy2 = map(int, (x1, y1, x2, y2))
            cx, cy = int((ix1 + ix2) / 2), int((iy1 + iy2) / 2)

            # فقط یک نشانگر کوچک اطراف بالن؛ خود بالن دست‌نخورده می‌ماند
            r = max(7, min(14, int(min(ix2 - ix1, iy2 - iy1) * 0.8)))
            cv2.circle(vis, (cx, cy), r, color, 2, cv2.LINE_AA)

            # نقطه‌ی مرکزی خیلی کوچک
            cv2.circle(vis, (cx, cy), 2, color, -1, cv2.LINE_AA)

            # نوشته‌ی درشت و خوانا
            label = f"ID {tid}  |  {dist_m:.0f}m" if dist_m else f"ID {tid}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.65
            thick = 2

            # نوشته کمی بالاتر از بالن
            tx, ty = cx + r + 6, max(cy - r, 25)

            # سایه مشکی برای خوانایی
            cv2.putText(vis, label, (tx + 2, ty + 2),
                        font, scale, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(vis, label, (tx, ty),
                        font, scale, color, thick, cv2.LINE_AA)

            # مسیر حرکت باریک
            for k in range(1, len(history)):
                p1 = tuple(map(int, history[k - 1]))
                p2 = tuple(map(int, history[k]))
                cv2.line(vis, p1, p2, color, 1, cv2.LINE_AA)




            
            csv_rows.append({
                "frame": frame_idx, "track_id": tid,
                "center_x_px": round(cx, 1), "center_y_px": round(cy, 1),
                "diameter_px": round(d_px, 1),
                "estimated_distance_m": round(dist_m, 1) if dist_m else "",
                "distance_uncertainty_pct": DIST_ACCURACY_PCT,
            })

        writer.write(vis)
        frame_times.append(time.time() - t0)
        prev_gray_blurred = gray_blurred

        if frame_idx % 10 == 0:
            print(f"فریم {frame_idx}: {len(tracked)} بالن ردیابی‌شده")

    cap.release()
    writer.release()

    # ذخیره‌ی لاگ CSV
    if csv_rows:
        with open(args.output_csv, "w", newline="", encoding="utf-8-sig") as f:
            wr = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            wr.writeheader()
            wr.writerows(csv_rows)

    avg_fps = 1 / (sum(frame_times) / len(frame_times)) if frame_times else 0
    print(f"\nتمام شد. {frame_idx} فریم پردازش شد.")
    print(f"میانگین FPS واقعی این اجرا: {avg_fps:.2f}")
    print(f"ویدیوی خروجی: {args.output_video}")
    print(f"لاگ CSV تشخیص‌ها: {args.output_csv}")


if __name__ == "__main__":
    main()