import os

# ضع مسارات المجلدات الموجودة على جهازك
images_dir = r'D:\KLYVERO_Advanced_YOLO\ultralytics\KLYVERO_Dataset\val\images'  # تأكد من اسم المجلد
labels_dir = r'D:\KLYVERO_Advanced_YOLO\ultralytics\KLYVERO_Dataset\val\labels'  # تأكد من اسم المجلد

deleted_count = 0

print("جاري تنظيف مجلد الصور...")
for img_file in os.listdir(images_dir):
    base_name, ext = os.path.splitext(img_file)
    
    # التأكد أن الملف صورة
    if ext.lower() not in ['.jpg', '.png', '.jpeg']:
        continue
        
    # بناء مسار ملف الإحداثيات المتوقع
    txt_path = os.path.join(labels_dir, base_name + '.txt')
    
    # إذا لم يكن ملف الـ txt موجوداً، احذف الصورة
    if not os.path.exists(txt_path):
        img_path = os.path.join(images_dir, img_file)
        os.remove(img_path)
        print(f"تم الحذف: {img_file}")
        deleted_count += 1

print(f"\nاكتمل التنظيف بنجاح! تم حذف {deleted_count} صورة ليس لها إحداثيات.")