import os
from ultralytics import YOLO

def auto_annotate_body(images_dir, output_labels_dir):
    os.makedirs(output_labels_dir, exist_ok=True)
    
    # استخدام الموديل القياسي (17 نقطة)
    model = YOLO('yolo11n-pose.pt') 
    
    print(f"بدء تحليل الصور في: {images_dir}")
    
    results = model.predict(source=images_dir, stream=True, conf=0.5)
    
    for result in results:
        image_name = os.path.splitext(os.path.basename(result.path))[0]
        txt_path = os.path.join(output_labels_dir, f"{image_name}.txt")
        
        if len(result.boxes) == 0 or result.keypoints is None:
            continue
            
        with open(txt_path, 'w') as f:
            for i in range(len(result.boxes)):
                class_id = int(result.boxes.cls[i].item())
                if class_id != 0:
                    continue
                
                box = result.boxes.xywhn[i].tolist()
                xyn = result.keypoints.xyn[i].tolist()   
                conf = result.keypoints.conf[i].tolist() 
                
                # كتابة الصندوق
                line = f"0 {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}"
                
                # دمج الـ 17 نقطة كاملة
                for (x, y), c in zip(xyn, conf):
                    line += f" {x:.6f} {y:.6f} {c:.6f}"
                    
                f.write(line + "\n")
                
    print(f"تم الانتهاء بنجاح! تم حفظ ملفات التدريب في: {output_labels_dir}")

auto_annotate_body(
    images_dir=r'D:\KLYVERO_Advanced_YOLO\ultralytics\YogaPoses\images',
    output_labels_dir=r'D:\KLYVERO_Advanced_YOLO\ultralytics\YogaPoses\labels'
)