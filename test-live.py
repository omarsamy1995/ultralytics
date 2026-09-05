import cv2
from ultralytics import YOLO

# تحميل أوزان KLYVERO
model = YOLO(r'D:\KLYVERO_Advanced_YOLO\ultralytics\best.pt')

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("تعذر الوصول للكاميرا.")
        break

    # حساب مساحة الإطار الإجمالية لفلتر الحجم
    frame_height, frame_width = frame.shape[:2]
    frame_area = frame_width * frame_height

    # تمرير الإطار للموديل مع فلتر الثقة الصارم (85%)
    # verbose=False لمنع طباعة التفاصيل في الـ Terminal وتخفيف الضغط
    results = model(frame, conf=0.80, verbose=False)
    
    # أخذ نسخة من الإطار للرسم عليها
    annotated_frame = frame.copy()

    for result in results:
        boxes = result.boxes
        keypoints = result.keypoints

        for i in range(len(boxes)):
            # استخراج أبعاد الصندوق
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            width = x2 - x1
            height = y2 - y1

            # 1. فلتر النسبة: تجاهل الكائنات العريضة (أكبر من طولها)
            if width > height:
                continue
                
            # 2. فلتر المساحة: الكائن يجب أن يشغل 15% على الأقل من الشاشة
            if (width * height) < (frame_area * 0.15):
                continue

            # رسم الصندوق المحيط باللون الأخضر للكائنات الصحيحة فقط
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # استخراج ورسم النقاط الـ 12 باللون الأحمر
            if keypoints is not None:
                # استخدام xy للحصول على الإحداثيات بالبيكسل الفعلي (وليس نسب مئوية)
                kpts = keypoints.xy[i].tolist() 
                
                for kx, ky in kpts:
                    # التأكد من أن النقطة ظاهرة داخل الإطار
                    if kx > 0 and ky > 0: 
                        cv2.circle(annotated_frame, (int(kx), int(ky)), 5, (0, 0, 255), -1)

    cv2.imshow("KLYVERO Live Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()