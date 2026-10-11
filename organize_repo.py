import os
import shutil

# ساختار پوشه‌های استاندارد
FOLDERS = ['assets', 'data_prep', 'training', 'inference', 'evaluation', 'weights', 'docs']

def organize_files():
    # 1. ساخت پوشه‌ها
    for folder in FOLDERS:
        os.makedirs(folder, exist_ok=True)
        print(f"Folder ready: {folder}/")

    # 2. قوانین جابجایی بر اساس پیشوند یا نام فایل
    files = [f for f in os.listdir('.') if os.path.isfile(f) and f.endswith('.py') and f != 'organize_repo.py']
    
    for file in files:
        target_folder = None
        
        # دسته‌بندی بر اساس شماره‌های فایل‌های شما
        try:
            prefix = int(file[:2])
            if 1 <= prefix <= 16 or prefix == 22:
                target_folder = 'data_prep'
            elif prefix in [17, 19, 30]:
                target_folder = 'training'
            elif prefix in [18, 20, 21, 25]:
                target_folder = 'evaluation'
            elif prefix in [23, 24, 26, 27, 28, 29]:
                target_folder = 'inference'
        except ValueError:
            pass

        # دسته‌بندی فایل‌های بدون شماره
        if file == 'final_inference.py':
            target_folder = 'inference'
        elif 'benchmark' in file.lower():
            target_folder = 'evaluation'

        # جابجایی
        if target_folder:
            shutil.move(file, os.path.join(target_folder, file))
            print(f"Moved: {file} --> {target_folder}/")
        else:
            print(f"Skipped (No rule matched): {file}")

if __name__ == "__main__":
    organize_files()
    print("\n--- مرتب‌سازی با موفقیت انجام شد ---")