from pathlib import Path

LABELS_DIR = Path(r"C:\BalloonProject\dataset_final\labels")
NUM_CLASSES = 1

errors = []
total_lines = 0

for split in ("train", "val", "test"):
    for txt_file in (LABELS_DIR / split).glob("*.txt"):
        for line_number, line in enumerate(txt_file.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            total_lines += 1
            parts = line.split()
            if len(parts) != 5:
                errors.append((split, txt_file.name, line_number, "expected 5 values"))
                continue
            class_id = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:])
            if not 0 <= class_id < NUM_CLASSES:
                errors.append((split, txt_file.name, line_number, "invalid class id"))
            if not all(0.0 <= v <= 1.0 for v in [x, y, w, h]):
                errors.append((split, txt_file.name, line_number, "coordinate out of [0,1]"))
            if w <= 0 or h <= 0:
                errors.append((split, txt_file.name, line_number, "non-positive box size"))

print(f"کل خط‌های بررسی‌شده: {total_lines}")
print(f"تعداد خطا: {len(errors)}")
for e in errors[:30]:
    print(e)