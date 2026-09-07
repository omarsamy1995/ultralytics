import os
import glob

# خريطة التصنيف الجديدة
class_map = {
    'front': '0',
    'back': '1',
    'side': '2',
    'full': '3'
}

# مسار مجلد الداتا سيت
dataset_path = r"D:\KLYVERO_Advanced_YOLO\ultralytics\KLYVERO_Dataset"
label_dirs = [os.path.join(dataset_path, "train", "labels"),
              os.path.join(dataset_path, "val", "labels"),
              os.path.join(dataset_path, "test", "labels")]

for label_dir in label_dirs:
    if not os.path.exists(label_dir):
        print(f"⚠️ المجلد غير موجود: {label_dir}")
        continue

    txt_files = glob.glob(os.path.join(label_dir, "*.txt"))
    for txt_file in txt_files:
        filename = os.path.basename(txt_file).lower()
        
        # البحث عن الكلمة المفتاحية في اسم الملف
        class_id = None
        for key, val in class_map.items():
            if key in filename:
                class_id = val
                break
                
        if class_id is None:
            continue
            
        # قراءة وتعديل الـ Class ID
        with open(txt_file, 'r') as f:
            lines = f.readlines()
            
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) > 0:
                parts[0] = class_id  # تغيير الـ Class
                new_lines.append(" ".join(parts) + "\n")
                
        with open(txt_file, 'w') as f:
            f.writelines(new_lines)

print("✅ تم تحديث جميع ملفات الـ Labels بنجاح!")

