from ultralytics import YOLO

# تحميل الموديل المطور بـ CBAM
model = YOLO('yolo11-cbam-pose.yaml')

# نقل الأوزان السابقة
results = model.train(
    data='/teamspace/studios/this_studio/ultralytics/KLYVERO_Dataset/data.yaml',
    pretrained='/teamspace/studios/this_studio/ultralytics/best.pt',  # مسار أفضل أوزان السابقة
    epochs=100,
    imgsz=640,
    batch=16,
    freeze=10,  # تجميد أول 10 طبقات للحفاظ على استخراج المعالم
    project='KLYVERO_Engine',
    name='yolo11n_CBAM_MultiTask'
)