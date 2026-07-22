import csv
import shutil
import subprocess
import sys
from pathlib import Path


ana_klasor = Path(__file__).resolve().parent

# Terminalden data.yaml yolu verilirse onu kullanır.
# Yol verilmezse varsayılan olarak COCO128 çalışır.
if len(sys.argv) > 1:
    dataset = str(Path(sys.argv[1]).expanduser().resolve())
else:
    dataset = "coco128.yaml"

# Her datasetin sonuçları farklı klasörde tutulur.
if dataset == "coco128.yaml":
    dataset_adi = "coco128"
else:
    dataset_yolu = Path(dataset)

    if dataset_yolu.stem.lower() in ["data", "dataset"]:
        dataset_adi = dataset_yolu.parent.name
    else:
        dataset_adi = dataset_yolu.stem

    dataset_adi = "".join(
        karakter if karakter.isalnum() or karakter in "-_" else "_"
        for karakter in dataset_adi
    )

sonuc_klasoru = ana_klasor / "results_by_dataset" / dataset_adi
ozet_dosyasi = sonuc_klasoru / "training_summary.csv"

official_yolo = ana_klasor / ".venv_auto_official/bin/yolo"
yolov13_yolo = ana_klasor / ".venv_auto_yolov13/bin/yolo"

epoch = 1
img_size = 640
batch = 2


modeller = []

for boyut in ["n", "s", "m", "l", "x"]:
    modeller.append(("YOLOv8", boyut, f"yolov8{boyut}.pt", "official", "mps"))

for boyut in ["t", "s", "m", "c", "e"]:
    modeller.append(("YOLOv9", boyut, f"yolov9{boyut}.pt", "official", "mps"))

for boyut in ["n", "s", "m", "b", "l", "x"]:
    modeller.append(("YOLOv10", boyut, f"yolov10{boyut}.pt", "official", "mps"))

for boyut in ["n", "s", "m", "l", "x"]:
    modeller.append(("YOLO11", boyut, f"yolo11{boyut}.pt", "official", "mps"))
    modeller.append(("YOLO12", boyut, f"yolo12{boyut}.pt", "official", "mps"))
    modeller.append(("YOLO26", boyut, f"yolo26{boyut}.pt", "official", "mps"))

for boyut in ["n", "s", "l", "x"]:
    modeller.append(("YOLOv13", boyut, f"yolov13{boyut}.yaml", "external", "cpu"))


eski_sonuc_klasorleri = [
    ana_klasor / "auto_registry_results",
    ana_klasor / "registry_training_results",
    ana_klasor / "official_yolo_model_results",
    ana_klasor / "all_yolo_version_results",
    ana_klasor / "all_yolo_version_results_rescue",
    ana_klasor / "all_model_results",
    ana_klasor / "version_model_results",
    ana_klasor / "external_yolov13_results",
    ana_klasor / "training_results",
    ana_klasor / "runs",
]

# Eski sonuç klasörleri yalnızca daha önce çalıştırılan COCO128 için kullanılır.
# Yeni datasetlerde başka datasetin ağırlıkları kopyalanmaz.
if dataset_adi == "coco128":
    eski_sonuc_klasorleri_aktif = eski_sonuc_klasorleri
else:
    eski_sonuc_klasorleri_aktif = []


def sonuc_var_mi(klasor):
    best_dosyasi = klasor / "weights/best.pt"
    return best_dosyasi.exists() and best_dosyasi.stat().st_size > 0


def eski_sonucu_bul(model_dosyasi):
    model_adi = Path(model_dosyasi).stem.lower()

    for ana_sonuc in eski_sonuc_klasorleri_aktif:
        if not ana_sonuc.exists():
            continue

        for best_dosyasi in ana_sonuc.rglob("weights/best.pt"):
            run_klasoru = best_dosyasi.parent.parent

            if model_adi in str(run_klasoru).lower():
                return run_klasoru

    return None


def sonucu_kopyala(eski_klasor, yeni_klasor):
    yeni_klasor.mkdir(parents=True, exist_ok=True)
    (yeni_klasor / "weights").mkdir(exist_ok=True)

    for dosya_adi in ["best.pt", "last.pt"]:
        kaynak = eski_klasor / "weights" / dosya_adi
        hedef = yeni_klasor / "weights" / dosya_adi

        if kaynak.exists():
            shutil.copy2(kaynak, hedef)

    for dosya in eski_klasor.iterdir():
        if dosya.is_file():
            shutil.copy2(dosya, yeni_klasor / dosya.name)


def modeli_egit(model_dosyasi, backend, cihaz, hedef_klasor):
    if backend == "external":
        yolo_komutu = yolov13_yolo
    else:
        yolo_komutu = official_yolo

    if not yolo_komutu.exists():
        return False, f"YOLO komutu bulunamadı: {yolo_komutu}"

    komut = [
        str(yolo_komutu),
        "detect",
        "train",
        f"model={model_dosyasi}",
        f"data={dataset}",
        f"epochs={epoch}",
        f"imgsz={img_size}",
        f"batch={batch}",
        f"device={cihaz}",
        "workers=0",
        f"project={hedef_klasor.parent}",
        f"name={hedef_klasor.name}",
        "exist_ok=True",
    ]

    if backend == "external":
        komut.append("amp=False")

    print("Komut:")
    print(" ".join(komut))

    sonuc = subprocess.run(komut, cwd=ana_klasor)

    if sonuc.returncode != 0:
        return False, f"Komut hata kodu: {sonuc.returncode}"

    if not sonuc_var_mi(hedef_klasor):
        return False, "best.pt oluşmadı"

    return True, ""


def main():
    if dataset != "coco128.yaml" and not Path(dataset).is_file():
        print("Dataset YAML dosyası bulunamadı:")
        print(dataset)
        return

    sonuc_klasoru.mkdir(parents=True, exist_ok=True)

    ozet = []
    hazir_model_sayisi = 0

    print("YOLO model eğitim kontrolü başladı")
    print("Dataset:", dataset)
    print("Sonuç klasörü:", sonuc_klasoru)
    print("Toplam model:", len(modeller))

    for sira, model in enumerate(modeller, start=1):
        aile, boyut, model_dosyasi, backend, cihaz = model

        model_adi = Path(model_dosyasi).stem
        hedef_klasor = sonuc_klasoru / aile.lower() / model_adi

        print()
        print("-" * 60)
        print(f"{sira}/{len(modeller)} - {aile} {boyut}")
        print("Model:", model_dosyasi)

        durum = ""
        hata = ""
        eski_sonuc_yolu = ""

        if sonuc_var_mi(hedef_klasor):
            durum = "mevcut"
            print("Daha önce final klasörüne eklenmiş")

        else:
            eski_sonuc = eski_sonucu_bul(model_dosyasi)

            if eski_sonuc:
                print("Önceki sonuç bulundu:", eski_sonuc)
                sonucu_kopyala(eski_sonuc, hedef_klasor)
                eski_sonuc_yolu = str(eski_sonuc)

                if sonuc_var_mi(hedef_klasor):
                    durum = "eski_sonuc_kopyalandi"
                else:
                    durum = "kopyalama_hatasi"
                    hata = "best.pt kopyalanamadı"

            else:
                print("Önceki sonuç bulunamadı, eğitim başlatılıyor")

                basarili, hata_mesaji = modeli_egit(
                    model_dosyasi,
                    backend,
                    cihaz,
                    hedef_klasor,
                )

                if basarili:
                    durum = "egitildi"
                else:
                    durum = "egitim_hatasi"
                    hata = hata_mesaji

        if sonuc_var_mi(hedef_klasor):
            hazir = "evet"
            hazir_model_sayisi += 1
        else:
            hazir = "hayir"

        print("Durum:", durum)
        print("Sonuç mevcut:", hazir)

        ozet.append({
            "model_ailesi": aile,
            "boyut": boyut,
            "model_dosyasi": model_dosyasi,
            "backend": backend,
            "cihaz": cihaz,
            "durum": durum,
            "sonuc_mevcut": hazir,
            "sonuc_klasoru": str(hedef_klasor),
            "eski_sonuc": eski_sonuc_yolu,
            "hata": hata,
        })

    with open(ozet_dosyasi, "w", newline="", encoding="utf-8") as dosya:
        alanlar = [
            "model_ailesi",
            "boyut",
            "model_dosyasi",
            "backend",
            "cihaz",
            "durum",
            "sonuc_mevcut",
            "sonuc_klasoru",
            "eski_sonuc",
            "hata",
        ]

        yazici = csv.DictWriter(dosya, fieldnames=alanlar)
        yazici.writeheader()
        yazici.writerows(ozet)

    print()
    print("=" * 60)
    print("İşlem tamamlandı")
    print("Toplam model:", len(modeller))
    print("Sonucu bulunan model:", hazir_model_sayisi)
    print("Eksik veya hatalı model:", len(modeller) - hazir_model_sayisi)
    print("Özet dosyası:", ozet_dosyasi)


if __name__ == "__main__":
    main()
