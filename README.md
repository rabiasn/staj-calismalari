# Staj Çalışmaları

Bu repository, staj süresince geliştirdiğim veri işleme ve nesne tespiti çalışmalarını içermektedir.

## Projeler

### COCO Annotation Validator

COCO formatındaki annotation dosyalarının kontrol edilmesi ve polygon segmentasyonların dikdörtgen bounding box yapısına dönüştürülmesi amacıyla hazırlanmıştır.

Dosya:

- `coco-annotation-validator/coco_tool.py`

### YOLO Model Training

Farklı YOLO detection model ailelerini aynı dataset üzerinde sırayla eğitmek ve sonuçları ayrı klasörlerde saklamak amacıyla hazırlanmıştır.

Çalıştırılan model aileleri:

- YOLOv8
- YOLOv9
- YOLOv10
- YOLO11
- YOLO12
- YOLOv13
- YOLO26

Dosyalar:

- `yolo-model-training/train_yolo_models.py`
- `yolo-model-training/training_summary.csv`
- `yolo-model-training/README.txt`

## Not

Model ağırlıkları, datasetler, sanal ortamlar ve büyük eğitim sonuçları dosya boyutları nedeniyle repository içerisine eklenmemiştir.
