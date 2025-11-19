# 🕵️ Deep Steganography Projesi

Bu proje, **Derin Öğrenme (Deep Learning)** tekniklerini kullanarak bir resmin içine başka bir resmi veya metni gizleyen ve daha sonra bu gizli veriyi geri çıkaran profesyonel bir uygulamadır.

Geleneksel steganografi yöntemlerinin aksine, bu proje 3 aşamalı bir **Konvolüsyonel Sinir Ağı (CNN)** kullanır. Bu sayede gizleme işlemi piksel seviyesinde çok daha karmaşık ve tespit edilmesi zor bir şekilde gerçekleşir.

## 🚀 Özellikler

*   **Gelişmiş Mimari:** Prep, Hiding ve Reveal olmak üzere 3 özel ağdan oluşur.
*   **Otomatik Veri Seti:** İnternetten eğitim için gerekli resimleri otomatik indirir.
*   **Özel Test Modu:** Kendi resimlerinizi (`kapak.jpg`, `gizli.jpg`) veya metinlerinizi (`gizli.txt`) kolayca saklayabilirsiniz.
*   **Metin-Resim Dönüşümü:** Yazılarınızı otomatik olarak resme çevirip saklar.
*   **Profesyonel Kod:** OOP (Nesne Yönelimli Programlama) ve Clean Code prensiplerine uygun yazılmıştır.

## 🛠️ Kurulum

Projeyi çalıştırmak için Python yüklü olmalıdır. Gerekli kütüphaneleri kurmak için terminale şu komutu yazın:

```bash
pip install tensorflow numpy matplotlib pillow requests
```

## ▶️ Kullanım

### 1. Standart Çalıştırma
Sadece aşağıdaki komutu yazın. Program otomatik olarak örnek resimler indirecek, modeli eğitecek ve bir demo yapacaktır.

```bash
python main.py
```

### 2. Kendi Resimlerinizi Saklama
Projenin olduğu klasöre (veya `resimler` klasörüne) şu dosyaları koyarsanız program otomatik olarak bunları kullanır:

*   **`kapak.jpg`**: İçine gizleme yapılacak ana resim.
*   **`gizli.jpg`**: Saklanacak olan gizli resim.
*   **`gizli.txt`**: Saklanacak olan gizli metin (Eğer resim yoksa bu kullanılır).

**Örnek:**
1.  Güzel bir manzara resmini `kapak.jpg` olarak kaydet.
2.  Saklamak istediğin şifreyi `gizli.txt` içine yaz.
3.  `python main.py` çalıştır.
4.  Sonuç `ozel_sonuc_metin.png` dosyasında belirecektir!

## 🧠 Nasıl Çalışır? (Teknik Detay)

Sistem 3 ana bileşenden oluşur:

1.  **Prep Network:** Gizli resmi (Secret) alır ve özelliklerini çıkararak saklanmaya uygun hale getirir.
2.  **Hiding Network (Encoder):** Kapak resmi (Cover) ile Prep çıktısını birleştirir. Sonuç olarak, içinde gizli mesaj olan ama dışarıdan bakınca kapak resminin aynısı gibi görünen **Container** resmini üretir.
3.  **Reveal Network (Decoder):** Sadece Container resmini alır ve içindeki gizli veriyi (Secret) geri inşa eder.

### Loss Fonksiyonu (Eğitim Mantığı)
Model şu iki hatayı aynı anda minimize etmeye çalışır:
*   `|| Kapak - Container ||`: Kapak resmi bozulmamalı (İnsan gözü fark etmemeli).
*   `|| Gizli - Ortaya Çıkan ||`: Gizli mesaj net bir şekilde geri okunabilmeli.

## 📂 Dosya Yapısı

*   `main.py`: Tüm proje kodlarını içeren ana dosya.
*   `resimler/`: İndirilen veya sizin eklediğiniz resimlerin bulunduğu klasör.
*   `sonuc.png`: Eğitim sonrası rastgele bir test sonucu.
*   `ozel_sonuc_*.png`: Sizin dosyalarınızla yapılan testlerin sonuçları.

---

