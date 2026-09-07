import cv2
import numpy as np
from ultralytics import YOLO

# 1. تحميل موديل الـ CBAM الممتاز الخاص بك
model = YOLO('/teamspace/studios/this_studio/ultralytics/runs/pose/KLYVERO_Engine/yolo11n_body_12kpt-2/weights/best.pt')

# 2. إعدادات فلتر التنعيم الزمني (Temporal Smoother)
# ألفا (Alpha) تتحكم في قوة التنعيم:
# 0.1 = تنعيم قوي جداً (حركة بطيئة وثابتة جداً)
# 0.9 = تنعيم ضعيف (استجابة سريعة جداً للحركة ولكن اهتزاز أعلى)
# 0.5 تعتبر نقطة توازن جيدة للـ Virtual Try-On
ALPHA = 0.5  
previous_keypoints = None

# فتح الكاميرا
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # إجراء التوقع
    results = model(frame, verbose=False)
    
    for r in results:
        # التأكد من وجود مفاصل تم التعرف عليها
        if r.keypoints is not None and len(r.keypoints.xy) > 0:
            # استخراج مصفوفة النقاط للإطار الحالي
            current_keypoints = r.keypoints.xy[0].cpu().numpy()
            
            # تطبيق فلتر الـ EMA إذا كان هناك نقاط سابقة
            if previous_keypoints is not None and current_keypoints.shape == previous_keypoints.shape:
                # المعادلة الرياضية للتنعيم: (الجديد * ألفا) + (القديم * (1 - ألفا))
                smoothed_keypoints = (current_keypoints * ALPHA) + (previous_keypoints * (1.0 - ALPHA))
            else:
                # إذا كانت هذه أول لقطة، لا تنعيم
                smoothed_keypoints = current_keypoints
            
            # حفظ النقاط المنعمة لتصبح "النقاط السابقة" في الإطار القادم
            previous_keypoints = smoothed_keypoints
            
            # --- رسم النقاط المنعمة يدوياً على الشاشة ---
            for kp in smoothed_keypoints:
                x, y = int(kp[0]), int(kp[1])
                if x > 0 and y > 0:  # رسم النقاط الظاهرة فقط
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    cv2.imshow("KLYVERO - Temporal Smoothed Try-On", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()