# -*- coding: utf-8 -*-
"""
Pamuk Yaprağı Hastalıklarını Sınıflandırma Modeli
(Orijinal + Çoğaltılmış Veri Seti ile)
"""

# 1. GEREKLİ KÜTÜPHANELER
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2  # OpenCV
import random
from tqdm import tqdm  # İlerleme çubuğu
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
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # DÜZELTİLDİ
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import shuffle  # YENİ EKLENDİ (Eğitim setini karıştırmak için)

# Sabit tohum değerleri (random state)
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# 2. VERİ SETİ YOLLARI VE PARAMETRELER
train_directory = "Cotton_Original_Dataset/"
aug_directory = "Cotton_Augmented_Dataset/"  # YENİ EKLENDİ
SIZE = 64
boyut = 64
NUM_CLASSES = 5

# Sınıf isimleri (Orijinal klasörler)
class_names_list = [
    "Alternaria Leaf Spot",
    "Bacterial Blight",
    "Fusarium Wilt",
    "Healthy Leaf",
    "Verticillium Wilt",
]

# Çoğaltılmış klasör isimleri (YENİ EKLENDİ)
aug_class_names_list = [
    "aug_Alternaria_Leaf",
    "aug_Bacterial_Blight",
    "aug_Fusarium_Wilt",
    "aug_Healthy_Leaf",
    "aug_Verticillium_Wilt",
]


# 3. YARDIMCI FONKSİYON: ARKA PLANI TEMİZLEME
# (Bu fonksiyonda değişiklik yok)
def removeBackground(orijinal_bgr):
    try:
        hsv = cv2.cvtColor(orijinal_bgr, cv2.COLOR_BGR2HSV)
        white_bg = 255 * np.ones_like(orijinal_bgr, dtype=np.uint8)
        gray = cv2.cvtColor(orijinal_bgr, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        x, y, w, h = cv2.boundingRect(thresh)
        ROI_original = orijinal_bgr[y : y + h, x : x + w]
        white_bg[y : y + h, x : x + w] = ROI_original
        return white_bg
    except Exception as e:
        return orijinal_bgr


# 4. VERİ YÜKLEME (BÖLÜM 1: ORİJİNAL VERİ SETİ)
veriseti = []
etiket = []

print("Orijinal Veri Seti Yükleniyor...")
for label, class_name in enumerate(class_names_list):
    class_path = os.path.join(train_directory, class_name)
    if not os.path.isdir(class_path):
        print(f"Uyarı: {class_path} klasörü bulunamadı.")
        continue
    images = os.listdir(class_path)
    print(f"Yükleniyor: {class_name} ({len(images)} resim)")
    for image_name in tqdm(images, desc=class_name):
        if image_name.startswith("."):
            continue
        image_path = os.path.join(class_path, image_name)
        image = cv2.imread(image_path)
        if image is None:
            continue

        image = removeBackground(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (boyut, boyut))
        veriseti.append(np.array(image))
        etiket.append(label)

print("Orijinal veri seti yüklendi.")

# 5. ORİJİNAL VERİYİ EĞİTİM VE TEST OLARAK AYIRMA
# (STRATEJİ DEĞİŞİKLİĞİ: Bu adım, çoğaltılmış veriyi eklemeden ÖNCE yapılır)
print("Orijinal veriler eğitim ve test olarak ayrılıyor...")

veriseti_np = np.array(veriseti)
etiket_np = np.array(etiket)
donusum = to_categorical(etiket_np, num_classes=NUM_CLASSES)

X_train, X_test, y_train, y_test = train_test_split(
    veriseti_np, donusum, test_size=0.20, random_state=42, stratify=donusum
)

# Hafızayı boşalt
del veriseti, etiket, veriseti_np, etiket_np, donusum
print(f"Orijinal eğitim seti boyutu: {X_train.shape}")
print(f"Test seti (sadece orijinallerden) boyutu: {X_test.shape}")


# 4.5 VERİ YÜKLEME (BÖLÜM 2: ÇOĞALTILMIŞ VERİ SETİ) (YENİ BÖLÜM)
veriseti_aug = []
etiket_aug = []

print("\nÇoğaltılmış (Augmented) Veri Seti Yükleniyor...")
for label, class_name in enumerate(aug_class_names_list):
    class_path = os.path.join(aug_directory, class_name)
    if not os.path.isdir(class_path):
        print(f"Uyarı: {class_path} klasörü bulunamadı.")
        continue
    images = os.listdir(class_path)
    print(f"Yükleniyor: {class_name} ({len(images)} resim)")
    for image_name in tqdm(images, desc=class_name):
        if image_name.startswith("."):
            continue
        image_path = os.path.join(class_path, image_name)
        image = cv2.imread(image_path)
        if image is None:
            continue

        # Aynı ön işlemleri uygula
        image = removeBackground(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (boyut, boyut))
        veriseti_aug.append(np.array(image))
        etiket_aug.append(label)  # Orijinal etiketle aynı (0-4)

print("Çoğaltılmış veri seti yüklendi.")

# 5.5 EĞİTİM VERİLERİNİ BİRLEŞTİRME (YENİ BÖLÜM)
if len(veriseti_aug) > 0:
    # Çoğaltılmış verileri numpy formatına çevir
    veriseti_aug_np = np.array(veriseti_aug)
    etiket_aug_np = np.array(etiket_aug)
    y_train_aug = to_categorical(etiket_aug_np, num_classes=NUM_CLASSES)

    # Orijinal eğitim seti ile çoğaltılmış veriyi birleştir
    X_train = np.concatenate((X_train, veriseti_aug_np), axis=0)
    y_train = np.concatenate((y_train, y_train_aug), axis=0)

    # Hafızayı boşalt
    del veriseti_aug, etiket_aug, veriseti_aug_np, etiket_aug_np, y_train_aug

    print("\nOrijinal ve Çoğaltılmış eğitim verileri birleştirildi.")
    # Modelin doğru öğrenmesi için birleştirilmiş eğitim setini karıştır
    print("Birleştirilmiş eğitim seti karıştırılıyor...")
    X_train, y_train = shuffle(X_train, y_train, random_state=42)
else:
    print("\nÇoğaltılmış veri bulunamadı, sadece orijinal eğitim verisi kullanılıyor.")

print(f"Toplam Eğitim Seti Boyutu (Orijinal+Aug): {X_train.shape}")
print(f"Test Seti Boyutu (Değişmedi): {X_test.shape}")


# 6. VERİ ÇOĞALTMA (DATA AUGMENTATION)
# (DEĞİŞİKLİK YOK - Bu hala çok önemli!)
# Bu jeneratör, birleştirilmiş X_train setini alıp HER EPOCH'ta
# yeniden rastgele değiştirerek modelin ezberlemesini engeller.
augment = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

# 7. CNN MODELİNİN TANIMLANMASI
# (Değişiklik yok)
print("CNN modeli oluşturuluyor...")
activation_function = (
    "relu"  # Aktivasyon fonksiyonunu seç relu, elu, silu, softplus, gelu, ...
)
INPUT_SHAPE = (SIZE, SIZE, 3)
cnn1 = Sequential()
cnn1.add(
    Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation=activation_function,
        input_shape=INPUT_SHAPE,
    )
)
cnn1.add(MaxPooling2D(pool_size=(2, 2)))
cnn1.add(BatchNormalization(axis=-1))
cnn1.add(Conv2D(filters=64, kernel_size=(3, 3), activation=activation_function))
cnn1.add(MaxPooling2D(pool_size=(2, 2)))
cnn1.add(BatchNormalization(axis=-1))
cnn1.add(Conv2D(filters=128, kernel_size=(3, 3), activation=activation_function))
cnn1.add(MaxPooling2D(pool_size=(2, 2)))
cnn1.add(Conv2D(filters=256, kernel_size=(3, 3), activation=activation_function))
cnn1.add(MaxPooling2D(pool_size=(2, 2)))
cnn1.add(Flatten())
cnn1.add(Dropout(0.4))
cnn1.add(
    Dense(
        256,
        activation=activation_function,
        kernel_regularizer=tf.keras.regularizers.l2(0.002),
    )
)
cnn1.add(Dense(NUM_CLASSES, activation="softmax"))
cnn1.summary()

# 8. MODELİ DERLEME VE EĞİTME
# (Değişiklik yok, ancak artık daha büyük olan X_train ile çalışacak)
print("\nModel derleniyor...")
cnn1.compile(
    optimizer="adam",  # Önceki önerilerdeki gibi learning_rate'i düşürebilirsiniz
    # optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), # İyileştirme için
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

batch_size = 32
epochs = 50  # Veri seti büyüdüğü için 50 epoch'ta kalması iyi bir başlangıç

print("Model (Orijinal+Augmented Veri) eğitiliyor...")
history = cnn1.fit(
    augment.flow(X_train, y_train, batch_size=batch_size),
    epochs=epochs,
    validation_data=(X_test, y_test),  # Test seti hala SADECE orijinallerden oluşuyor
    steps_per_epoch=len(X_train) // batch_size,
)
print("Eğitim tamamlandı.")

# 9. MODELİ DEĞERLENDİRME VE SONUÇLARI GÖRSELLEŞTİRME
# (Değişiklik yok)
print("\nTest seti üzerinde model değerlendiriliyor:")
loss, accuracy = cnn1.evaluate(X_test, y_test)
print(f"Test Kaybı (Loss): {loss:.4f}")
print(f"Test Başarısı (Accuracy): {accuracy:.4f}")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Eğitim Başarısı")
plt.plot(history.history["val_accuracy"], label="Doğrulama Başarısı")
plt.title("Model Başarısı")
plt.xlabel("Epoch")
plt.ylabel("Başarı (Accuracy)")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Eğitim Kaybı")
plt.plot(history.history["val_loss"], label="Doğrulama Kaybı")
plt.title("Model Kaybı (Loss)")
plt.xlabel("Epoch")
plt.ylabel("Kayıp (Loss)")
plt.legend()
plt.tight_layout()
plt.show()

y_pred_prob = cnn1.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)
print("\nSınıflandırma Raporu:")
print(classification_report(y_true, y_pred, target_names=class_names_list))

# 10. MODELİ KAYDETME (YENİ BÖLÜM)
model_kayit_adi = f"cotton_model_{activation_function}.keras"
print(f"\nModel {model_kayit_adi} olarak kaydediliyor...")
cnn1.save(model_kayit_adi)
print(f"Model başarıyla {model_kayit_adi} klasörüne kaydedildi.")
