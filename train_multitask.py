from ultralytics import YOLO

# تحميل الموديل المطور بـ CBAM
model = YOLO('/teamspace/studios/this_studio/ultralytics/ultralytics/cfg/models/11/yolo11-cbam-pose.yaml')

# نقل الأوزان السابقة
results = model.train(
    data='/teamspace/studios/this_studio/ultralytics/KLYVERO_Dataset/data.yaml',
    pretrained='/teamspace/studios/this_studio/ultralytics/runs/pose/KLYVERO_Engine/yolo11n_body_12kpt-2/weights/best.pt',  # مسار أفضل أوزان السابقة
    epochs=100,
    imgsz=640,
    batch=16,
    freeze=10,  # تجميد أول 10 طبقات للحفاظ على استخراج المعالم
    project='KLYVERO_Engine',
    name='yolo11n_CBAM_MultiTask'
)