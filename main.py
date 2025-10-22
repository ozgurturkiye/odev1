# -*- coding: utf-8 -*-
"""
Pamuk Yaprağı Hastalıklarını Sınıflandırma Modeli
@author: Sizin Adınız (veya @author: IIcetiner - orijinalden uyarlandı)
"""

# 1. GEREKLİ KÜTÜPHANELER
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2  # OpenCV (Görüntü işleme için)
import random
from tqdm import tqdm  # İşlem ilerlemesini gösteren çubuk için
from PIL import Image

# Makine Öğrenimi ve Derin Öğrenme Kütüphaneleri
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dropout,
    BatchNormalization,
)
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# Sabit tohum değerleri (random state)
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# 2. VERİ SETİ YOLLARI VE PARAMETRELER
train_directory = "Cotton_Original_Dataset/"
SIZE = 64  # Resimlerin yeniden boyutlandırılacağı boyut (64x64)
boyut = 64
NUM_CLASSES = 5  # Sınıf sayısı (5 adet)

# Sınıf isimleri (klasör isimleriyle aynı olmalı)
class_names_list = [
    "Alternaria Leaf Spot",
    "Bacterial Blight",
    "Fusarium Wilt",
    "Healthy Leaf",
    "Verticillium Wilt",
]


# 3. YARDIMCI FONKSİYON: ARKA PLANI TEMİZLEME
# Orijinal koddaki fonksiyon, BGR formatında bir resim alıp
# arka planı beyazlatılmış BGR bir resim döndürecek şekilde düzeltildi.
def removeBackground(orijinal_bgr):
    try:
        # BGR'den HSV renk uzayına geç
        hsv = cv2.cvtColor(orijinal_bgr, cv2.COLOR_BGR2HSV)

        # Arka plan için beyaz bir BGR tuval oluştur
        white_bg = 255 * np.ones_like(orijinal_bgr, dtype=np.uint8)

        # Maske oluşturmak için gri tonlamalı resim kullan
        gray = cv2.cvtColor(orijinal_bgr, cv2.COLOR_BGR2GRAY)
        # Eşikleme (Thresholding)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # Resimdeki ana nesnenin (yaprak) sınırlarını bul
        x, y, w, h = cv2.boundingRect(thresh)

        # Orijinal BGR resimden sadece yaprağın olduğu bölgeyi (ROI) al
        ROI_original = orijinal_bgr[y : y + h, x : x + w]

        # Beyaz arka planın üzerine sadece bu yaprak bölgesini yerleştir
        white_bg[y : y + h, x : x + w] = ROI_original
        return white_bg
    except Exception as e:
        # print(f"Arka plan temizleme hatası: {e}")
        # Hata olursa orijinal resmi döndür
        return orijinal_bgr


# 4. VERİ YÜKLEME VE ÖN İŞLEME
veriseti = []
etiket = []

print("Veri seti yükleniyor...")

for label, class_name in enumerate(class_names_list):
    class_path = os.path.join(train_directory, class_name)

    # Eğer klasör yoksa uyar
    if not os.path.isdir(class_path):
        print(f"Uyarı: {class_path} klasörü bulunamadı.")
        continue

    images = os.listdir(class_path)
    print(f"Yükleniyor: {class_name} ({len(images)} resim)")

    # tqdm ile ilerleme çubuğu ekle
    for image_name in tqdm(images, desc=class_name):
        # .DS_Store gibi sistem dosyalarını atla
        if image_name.startswith("."):
            continue

        image_path = os.path.join(class_path, image_name)

        # Resmi BGR olarak oku (removeBackground BGR bekliyor)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Uyarı: {image_path} okunamadı, atlanıyor.")
            continue

        # 1. Arka planı temizle
        image = removeBackground(image)

        # 2. Modeller genelde RGB ile eğitilir (BGR -> RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 3. Yeniden boyutlandır (Tüm resimler aynı boyutta olmalı)
        image = cv2.resize(image, (boyut, boyut))

        # İşlenmiş resmi 'veriseti' listesine ekle
        veriseti.append(np.array(image))
        # Etiketini (0, 1, 2, 3, 4) 'etiket' listesine ekle
        etiket.append(label)

print("Veri seti yüklendi.")


# 5. VERİYİ EĞİTİM VE TEST OLARAK AYIRMA
print("Veriler eğitim ve test olarak ayrılıyor...")

# Listeleri numpy dizilerine çevir
veriseti_np = np.array(veriseti)
etiket_np = np.array(etiket)

# Etiketleri kategorik formata çevir (One-Hot Encoding)
# Örn: 2 -> [0, 0, 1, 0, 0]
donusum = to_categorical(etiket_np, num_classes=NUM_CLASSES)

# Veriyi %80 eğitim, %20 test olarak ayır
X_train, X_test, y_train, y_test = train_test_split(
    veriseti_np, donusum, test_size=0.20, random_state=42, stratify=donusum
)  # Sınıf dağılımını koru

# Hafızayı boşaltmak için büyük listeleri sil
del veriseti, etiket, veriseti_np, etiket_np

print(f"Eğitim seti boyutu: {X_train.shape}")
print(f"Test seti boyutu: {X_test.shape}")


# 6. VERİ ÇOĞALTMA (DATA AUGMENTATION)
# Orijinal kodda bu jeneratörler tanımlanmış.
# Bu jeneratör, model.fit içinde veriyi "anlık" olarak çoğaltmak için kullanılır.
augment = ImageDataGenerator(
    rotation_range=25,  # Rastgele döndürme
    width_shift_range=0.1,  # Yatay kaydırma
    height_shift_range=0.1,  # Dikey kaydırma
    shear_range=0.2,  # Eğme/Bükme
    zoom_range=0.2,  # Yakınlaştırma
    horizontal_flip=True,  # Yatayda ters çevirme
    fill_mode="nearest",
)

# 7. CNN MODELİNİN TANIMLANMASI (İsteğiniz üzerine düzeltilmiş hali)
print("CNN modeli oluşturuluyor...")

INPUT_SHAPE = (SIZE, SIZE, 3)

cnn1 = Sequential()

# Katman 1
cnn1.add(
    Conv2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=INPUT_SHAPE)
)
cnn1.add(MaxPooling2D(pool_size=(2, 2)))
cnn1.add(BatchNormalization(axis=-1))

# Katman 2
cnn1.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
cnn1.add(MaxPooling2D(pool_size=(2, 2)))
cnn1.add(BatchNormalization(axis=-1))

# Katman 3
cnn1.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu"))
cnn1.add(MaxPooling2D(pool_size=(2, 2)))

# Katman 4
cnn1.add(Conv2D(filters=256, kernel_size=(3, 3), activation="relu"))
cnn1.add(MaxPooling2D(pool_size=(2, 2)))

# Karar verme katmanları
cnn1.add(Flatten())  # 2D veriyi 1D'ye düzleştir
cnn1.add(Dropout(0.4))  # Ezberlemeyi (overfitting) önlemek için
cnn1.add(
    Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.002))
)

# ÇIKIŞ KATMANI
# Orijinal kodda 4 sınıf vardı, sizin 5 sınıfınız var. Burası 5 olarak güncellendi.
cnn1.add(Dense(NUM_CLASSES, activation="softmax"))  # 5 sınıf için olasılık çıktısı

# Modelin özetini göster
cnn1.summary()


# 8. MODELİ DERLEME VE EĞİTME (Bu kısım orijinal kodda eksikti)

print("\nModel derleniyor...")
cnn1.compile(
    optimizer="adam",  # Popüler bir optimize edici
    loss="categorical_crossentropy",  # Çoklu sınıflandırma için kayıp fonksiyonu
    metrics=["accuracy"],  # Başarı metriği
)

batch_size = 32
epochs = 50  # Eğitim döngüsü sayısı (deneyerek artırılabilir)

print("Model eğitiliyor...")

# Veri çoğaltma (augmentation) kullanarak modeli eğit
history = cnn1.fit(
    augment.flow(X_train, y_train, batch_size=batch_size),
    epochs=epochs,
    validation_data=(X_test, y_test),  # Test seti ile doğrula
    steps_per_epoch=len(X_train) // batch_size,
)

print("Eğitim tamamlandı.")


# 9. MODELİ DEĞERLENDİRME VE SONUÇLARI GÖRSELLEŞTİRME

print("\nTest seti üzerinde model değerlendiriliyor:")
loss, accuracy = cnn1.evaluate(X_test, y_test)
print(f"Test Kaybı (Loss): {loss:.4f}")
print(f"Test Başarısı (Accuracy): {accuracy:.4f}")

# Eğitim ve Doğrulama grafikleri
plt.figure(figsize=(12, 5))

# Başarı (Accuracy) grafiği
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Eğitim Başarısı")
plt.plot(history.history["val_accuracy"], label="Doğrulama Başarısı")
plt.title("Model Başarısı")
plt.xlabel("Epoch")
plt.ylabel("Başarı (Accuracy)")
plt.legend()

# Kayıp (Loss) grafiği
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Eğitim Kaybı")
plt.plot(history.history["val_loss"], label="Doğrulama Kaybı")
plt.title("Model Kaybı (Loss)")
plt.xlabel("Epoch")
plt.ylabel("Kayıp (Loss)")
plt.legend()

plt.tight_layout()
plt.show()

# Sınıflandırma Raporu
y_pred_prob = cnn1.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)

print("\nSınıflandırma Raporu:")
print(classification_report(y_true, y_pred, target_names=class_names_list))
