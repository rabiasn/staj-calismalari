YOLO MODEL EĞİTİM ÇALIŞMASI

Bu script, mevcut YOLO detection model ailelerinin farklı
varyantlarını aynı dataset üzerinde sırayla eğitmek ve sonuçları
ayrı klasörlere kaydetmek amacıyla hazırlanmıştır.

Çalıştırılan modeller:

- YOLOv8: n, s, m, l, x
- YOLOv9: t, s, m, c, e
- YOLOv10: n, s, m, b, l, x
- YOLO11: n, s, m, l, x
- YOLO12: n, s, m, l, x
- YOLO26: n, s, m, l, x
- YOLOv13: n, s, l, x

Toplam model sayısı: 35

Son çalıştırma sonucu:

- Sonucu bulunan model: 35
- Eksik veya hatalı model: 0

V14-V25 isimleri, kurulu Ultralytics paketindeki model
mimarisi dosyaları arasında bulunmadığı için yapay model adı
üretilerek çalışmaya eklenmemiştir.

KULLANIM

1. Varsayılan COCO128 datasetiyle çalıştırma:

python3 train_yolo_models.py

2. Farklı bir YOLO detection datasetiyle çalıştırma:

python3 train_yolo_models.py "/tam/yol/data.yaml"

Örnek:

python3 train_yolo_models.py "/Users/rabia/Desktop/ucak_dataset/data.yaml"

Script proje klasöründe bırakılarak terminalde farklı bir klasörden
tam dosya yolu ile de çalıştırılabilir:

python3 /Users/rabia/Desktop/STAJ/yolo_training/train_yolo_models.py data.yaml

Her datasetin sonuçları ayrı tutulur:

results_by_dataset/<dataset_adi>/

Datasetin YOLO object detection formatında olması gerekir:

dataset/
  images/
    train/
    val/
  labels/
    train/
    val/
  data.yaml

NOTLAR

- Script tek bir datasete bağlı değildir.
- Çalıştırma sırasında data.yaml yolu parametre olarak verilebilir.
- Resmi Ultralytics modelleri Apple MPS üzerinde çalıştırılır.
- Harici YOLOv13 modelleri uyumluluk nedeniyle CPU üzerinde çalıştırılır.
- Epoch, batch ve görüntü boyutu script içinden değiştirilebilir.
- Eğitim ağırlıkları dosya boyutları yüksek olduğu için teslim
  paketine eklenmemiştir.
