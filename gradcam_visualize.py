# -*- coding: utf-8 -*-
"""
Grad-CAM Görselleştirme - Pamuk Yaprak Hastalıkları
Model: cotton_model_relu.keras
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Sınıf isimleri
class_names = [
    "Alternaria Leaf Spot",
    "Bacterial Blight",
    "Fusarium Wilt",
    "Healthy Leaf",
    "Verticillium Wilt",
]

SIZE = 64


def removeBackground(orijinal_bgr):
    """Arka planı temizle"""
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


def load_and_preprocess_image(img_path, target_size=(64, 64)):
    """Görüntüyü yükle ve ön işleme uygula"""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Görüntü yüklenemedi: {img_path}")

    # Arka plan temizleme
    img = removeBackground(img)
    # BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Yeniden boyutlandır
    img = cv2.resize(img, target_size)

    return img


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Grad-CAM ısı haritası oluştur

    Args:
        img_array: Giriş görüntüsü (batch_size, height, width, channels)
        model: Eğitilmiş model
        last_conv_layer_name: Son konvolüsyon katmanının adı
        pred_index: Hedef sınıf indeksi (None ise en yüksek tahmin)

    Returns:
        heatmap: Normalize edilmiş ısı haritası (0-1 arası)
    """
    # Grad-CAM modeli oluştur
    grad_model = Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )

    # Gradyanları hesapla
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Konvolüsyon çıktısına göre gradyanlar
    grads = tape.gradient(class_channel, conv_outputs)

    # Global average pooling - her kanal için ortalama gradyan
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Konvolüsyon çıktılarını gradyanlarla ağırlıklandır
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize et (ReLU uygula - negatif değerleri sıfırla)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()


def apply_heatmap_on_image(img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Isı haritasını görüntü üzerine uygula

    Args:
        img: Orijinal görüntü (RGB, 0-255)
        heatmap: Isı haritası (0-1 arası)
        alpha: Şeffaflık oranı
        colormap: Renk haritası

    Returns:
        superimposed_img: Birleştirilmiş görüntü
    """
    # Isı haritasını görüntü boyutuna getir
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # 0-255 aralığına çevir
    heatmap = np.uint8(255 * heatmap)

    # Renk haritası uygula
    heatmap_colored = cv2.applyColorMap(heatmap, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Orijinal görüntü ile birleştir
    superimposed_img = heatmap_colored * alpha + img * (1 - alpha)
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    return superimposed_img, heatmap


def visualize_gradcam(
    img_path, model, last_conv_layer_name, class_names, save_path=None
):
    """
    Grad-CAM görselleştirmesi yap ve kaydet
    """
    # Görüntüyü yükle
    img = load_and_preprocess_image(img_path, target_size=(SIZE, SIZE))

    # Model için hazırla
    img_array = np.expand_dims(img, axis=0)  # Batch boyutu ekle

    # Tahmin yap
    predictions = model.predict(img_array, verbose=0)
    pred_class = np.argmax(predictions[0])
    pred_prob = predictions[0][pred_class]

    print(f"Tahmin: {class_names[pred_class]} ({pred_prob*100:.2f}%)")
    print(f"Tüm sınıf olasılıkları:")
    for i, prob in enumerate(predictions[0]):
        print(f"  {class_names[i]}: {prob*100:.2f}%")

    # Grad-CAM hesapla
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_class)

    # Isı haritasını görüntü üzerine uygula
    superimposed_img, heatmap_colored = apply_heatmap_on_image(img, heatmap, alpha=0.4)

    # Görselleştir
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Orijinal görüntü
    axes[0].imshow(img)
    axes[0].set_title("Orijinal Görüntü", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # Isı haritası (sadece)
    im = axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Isı Haritası", fontsize=12, fontweight="bold")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Birleştirilmiş görüntü
    axes[2].imshow(superimposed_img)
    axes[2].set_title("Grad-CAM Overlay", fontsize=12, fontweight="bold")
    axes[2].axis("off")

    # Tahmin sonuçları
    axes[3].barh(class_names, predictions[0])
    axes[3].set_xlabel("Olasılık", fontsize=10)
    axes[3].set_title("Sınıf Tahminleri", fontsize=12, fontweight="bold")
    axes[3].set_xlim([0, 1])

    # En yüksek tahmini vurgula
    axes[3].get_children()[pred_class].set_color("red")

    plt.tight_layout()

    # Kaydet
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\nGörselleştirme kaydedildi: {save_path}")

    plt.show()

    return heatmap, superimposed_img, predictions[0]


def batch_gradcam_visualization(
    test_folder,
    model,
    last_conv_layer_name,
    class_names,
    num_samples=5,
    output_folder="gradcam_results",
):
    """
    Bir klasördeki birden fazla görüntü için Grad-CAM görselleştirmesi
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Klasördeki tüm görüntüleri al
    image_files = [
        f
        for f in os.listdir(test_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    # Rastgele örnek seç
    if len(image_files) > num_samples:
        image_files = np.random.choice(image_files, num_samples, replace=False)

    print(f"\n{len(image_files)} görüntü için Grad-CAM oluşturuluyor...\n")

    for img_file in image_files:
        img_path = os.path.join(test_folder, img_file)
        save_path = os.path.join(output_folder, f"gradcam_{img_file}")

        print(f"\n{'='*60}")
        print(f"İşleniyor: {img_file}")
        print("=" * 60)

        try:
            visualize_gradcam(
                img_path, model, last_conv_layer_name, class_names, save_path=save_path
            )
        except Exception as e:
            print(f"Hata: {img_file} işlenirken sorun oluştu: {e}")


# ============================================================================
# ANA PROGRAM
# ============================================================================

if __name__ == "__main__":

    # Modeli yükle
    print("Model yükleniyor...")
    model_path = "cotton_model_relu.keras"
    model = load_model(model_path)
    print("Model başarıyla yüklendi!\n")

    # Model mimarisini göster
    print("Model Katmanları:")
    for i, layer in enumerate(model.layers):
        print(f"{i}: {layer.name} - {layer.__class__.__name__}")

    # Son konvolüsyon katmanını belirle
    # Modelinize göre bu katman adını değiştirmeniz gerekebilir
    last_conv_layer_name = "conv2d_3"  # 4. Conv2D katmanı (256 filtreli)

    print(f"\nKullanılan Grad-CAM katmanı: {last_conv_layer_name}")
    print("=" * 60)

    # TEK GÖRÜNTÜ ÖRNEĞİ
    # Kendi görüntü yolunuzu buraya yazın
    img_path = "./Cotton_Original_Dataset/Bacterial Blight/bacterial_(33).png"  # KENDI DOSYA YOLUNUZU YAZIN

    if os.path.exists(img_path):
        print(f"\nTek görüntü için Grad-CAM oluşturuluyor: {img_path}")
        visualize_gradcam(
            img_path,
            model,
            last_conv_layer_name,
            class_names,
            save_path="gradcam_single_result.png",
        )
    else:
        print(f"\nUyarı: {img_path} bulunamadı. Lütfen geçerli bir görüntü yolu girin.")

    # TOPLU GÖRÜNTÜ İŞLEME (opsiyonel)
    # Bir test klasörü varsa burayı kullanabilirsiniz
    test_folder = "test_images"  # KENDI KLASÖR YOLUNUZU YAZIN

    if os.path.exists(test_folder) and os.path.isdir(test_folder):
        print(f"\n\nKlasör için toplu Grad-CAM oluşturuluyor: {test_folder}")
        batch_gradcam_visualization(
            test_folder, model, last_conv_layer_name, class_names, num_samples=5
        )
    else:
        print(f"\nBilgi: {test_folder} klasörü bulunamadı. Toplu işleme atlanıyor.")

    print("\n" + "=" * 60)
    print("İşlem tamamlandı!")
    print("=" * 60)
