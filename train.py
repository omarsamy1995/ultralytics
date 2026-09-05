from ultralytics import YOLO

# 1. تهيئة المعمارية الهندسية الجديدة
model = YOLO('yolo11-cbam-pose.yaml')

# 2. حقن المعرفة البصرية الجاهزة للطبقات المتطابقة (Transfer Learning)
model.load('D:\KLYVERO_Advanced_YOLO\ultralytics\runs\pose\KLYVERO_Engine\yolo11n_body_12kpt-2\weights\best.pt')

# 3. بدء التدريب 
results = model.train(
    data='D:\KLYVERO_Advanced_YOLO\ultralytics\KLYVERO_Dataset\data.yaml',
    epochs=300,
    imgsz=640,
    device=0,
    batch=16,
  
    freeze=10,                   
    lr0=0.001,                   
    patience=30,                 
    weight_decay=0.0005,         
    
    # التعزيز البصري المخصص لوقوف الأشخاص
    degrees=10.0,                
    scale=0.2,                   
    hsv_s=0.7,                   
    hsv_v=0.4,                   
    close_mosaic=15,             
    
    project='KLYVERO_Engine',
    name='yolo11n_CBAM_body'
)