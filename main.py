import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
import numpy as np
import matplotlib.pyplot as plt
import requests
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import os

# ==========================================
# AYARLAR (KONFIGURASYON)
# ==========================================
IMG_SIZE = (128, 128)  # Eğitim hızı için 128x128 ideal
EPOCHS = 100           # Eğitim döngüsü sayısı (Daha iyi sonuç için artırılabilir)
BATCH_SIZE = 8         # Her adımda işlenen resim sayısı
BETA = 1.0             # Gizli resim hatasının ağırlığı (1.0 = eşit önem)

print("Program başlatılıyor...")
print(f"TensorFlow Versiyonu: {tf.__version__}")

# ==========================================
# 1. VERİ HAZIRLAMA (GÜNCELLENDİ)
# ==========================================
def download_and_save_images(folder_name="resimler"):
    """
    İnternetten güzel manzara/tablo resimleri indirip klasöre kaydeder.
    Eğer klasörde zaten resim varsa indirme yapmaz, onları kullanır.
    """
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"'{folder_name}' klasörü oluşturuldu.")
    
    # Klasördeki mevcut resimleri kontrol et
    existing_files = [f for f in os.listdir(folder_name) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(existing_files) >= 2:
        print(f"'{folder_name}' klasöründe {len(existing_files)} resim bulundu. Bunlar kullanılacak.")
        return

    print("Klasör boş veya yetersiz. İnternetten örnek resimler indiriliyor...")
    
    # Daha güvenilir ve güzel resim kaynakları (Manzara, Tablo vb.)
    urls = [
        ("manzara.jpg", "https://images.unsplash.com/photo-1472214103451-9374bd1c798e?q=80&w=400&auto=format&fit=crop"), # Doğa
        ("tablo.jpg", "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg/402px-Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg"), # Mona Lisa
        ("sehir.jpg", "https://images.unsplash.com/photo-1519501025264-65ba15a82390?q=80&w=400&auto=format&fit=crop"), # Şehir
        ("kedi.jpg", "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?q=80&w=400&auto=format&fit=crop")  # Kedi
    ]
    
    headers = {'User-Agent': 'Mozilla/5.0'} # Bazı siteler botları engeller, tarayıcı gibi görünelim
    
    for filename, url in urls:
        path = os.path.join(folder_name, filename)
        if not os.path.exists(path):
            try:
                response = requests.get(url, headers=headers, timeout=10)
                if response.status_code == 200:
                    with open(path, 'wb') as f:
                        f.write(response.content)
                    print(f"İndirildi: {filename}")
                else:
                    print(f"İndirilemedi (Status {response.status_code}): {url}")
            except Exception as e:
                print(f"Hata ({url}): {e}")

def load_data(folder_name="resimler"):
    """Belirtilen klasördeki tüm resimleri okur ve hazırlar."""
    images = []
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif')
    
    # Önce indirmeyi/kontrolü yap
    download_and_save_images(folder_name)
    
    print("Resimler yükleniyor...")
    files = [f for f in os.listdir(folder_name) if f.lower().endswith(valid_extensions)]
    
    for filename in files:
        path = os.path.join(folder_name, filename)
        try:
            img = Image.open(path).convert('RGB')
            img = img.resize(IMG_SIZE)
            img_array = np.array(img) / 255.0
            images.append(img_array)
        except Exception as e:
            print(f"Okunamadı: {filename} - {e}")
            
    if not images:
        print("HATA: Hiçbir resim yüklenemedi! Lütfen 'resimler' klasörüne .jpg veya .png dosyaları koyun.")
        # Fallback yine de kod çökmesin diye
        return np.random.rand(10, *IMG_SIZE, 3), np.random.rand(10, *IMG_SIZE, 3)

    # Veri çoğaltma (Data Augmentation)
    # Elimizdeki az sayıda resmi kopyalayarak veri setini büyütüyoruz
    X_train = np.array(images * 50) 
    np.random.shuffle(X_train)
    
    # Cover ve Secret olarak ayır
    split = len(X_train) // 2
    covers = X_train[:split]
    secrets = X_train[split:2*split]
    
    # Boyut eşitleme
    min_len = min(len(covers), len(secrets))
    return covers[:min_len], secrets[:min_len]

# ==========================================
# 2. MODEL MİMARİSİ (3 AŞAMALI AĞ)
# ==========================================
def make_model():
    """Deep Steganography Modelini Oluşturur."""
    
    # --- Girdiler ---
    input_cover = layers.Input(shape=(*IMG_SIZE, 3), name='input_cover')
    input_secret = layers.Input(shape=(*IMG_SIZE, 3), name='input_secret')
    
    # --- 1. Prep Network (Gizli Resmi Hazırlama) ---
    # Gizli resmin özelliklerini çıkarır ve saklanmaya uygun hale getirir
    x = layers.Conv2D(50, (3, 3), padding='same', activation='relu')(input_secret)
    x = layers.Conv2D(50, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(50, (3, 3), padding='same', activation='relu')(x)
    prep_output = layers.Conv2D(50, (3, 3), padding='same', activation='relu')(x)
    
    # --- 2. Hiding Network (Encoder) ---
    # Kapak resmi ile Prep çıktısını birleştirir
    concat_input = layers.Concatenate()([input_cover, prep_output])
    
    x = layers.Conv2D(50, (3, 3), padding='same', activation='relu')(concat_input)
    x = layers.Conv2D(50, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(50, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(50, (3, 3), padding='same', activation='relu')(x)
    
    # Çıktı: Container Image (3 kanal RGB)
    # Aktivasyon 'sigmoid' çünkü çıktı 0-1 arasında olmalı (Piksel değerleri)
    container_output = layers.Conv2D(3, (3, 3), padding='same', activation='sigmoid', name='output_container')(x)
    
    # --- 3. Reveal Network (Decoder) ---
    # Sadece Container resmine bakarak gizli resmi bulmaya çalışır
    x = layers.Conv2D(50, (3, 3), padding='same', activation='relu')(container_output)
    x = layers.Conv2D(50, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(50, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(50, (3, 3), padding='same', activation='relu')(x)
    
    # Çıktı: Revealed Secret (3 kanal RGB)
    reveal_output = layers.Conv2D(3, (3, 3), padding='same', activation='sigmoid', name='output_reveal')(x)
    
    # Modeli birleştir
    model = models.Model(inputs=[input_cover, input_secret], outputs=[container_output, reveal_output])
    return model

# ==========================================
# 3. ÖZEL LOSS FONKSİYONU VE DERLEME
# ==========================================
def custom_loss(beta=1.0):
    """
    Toplam Hata = ||Kapak - Container|| + beta * ||Gizli - OrtayaÇıkan||
    """
    def loss_function(y_true, y_pred):
        # Keras model.compile'da loss dictionary olarak verilecek,
        # ama genel mantığı burada anlamak önemli.
        # Biz aşağıda compile ederken standart MSE kullanıp ağırlıklandıracağız.
        pass
    return loss_function

# ==========================================
# 5. ÖZEL TEST VE METİN SAKLAMA MODU
# ==========================================
def text_to_image(text):
    """Metni resme çevirir (Siyah üzerine beyaz yazı)."""
    img = Image.new('RGB', IMG_SIZE, color=(0, 0, 0))
    d = ImageDraw.Draw(img)
    # Font yüklemeye çalış, yoksa varsayılanı kullan
    try:
        # Windows varsayılan fontu
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    # Metni yaz (Basitçe sol üst köşeye)
    d.text((10, 50), text, fill=(255, 255, 255), font=font)
    return img

def prepare_single_image(image_path):
    """Tek bir resmi model için hazırlar."""
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(IMG_SIZE)
        return np.array(img) / 255.0
    except Exception as e:
        print(f"Hata: {image_path} açılamadı. {e}")
        return None

def run_custom_test(model):
    print("\n" + "="*40)
    print("      ÖZEL TEST MODU (NE SAKLADIK?)")
    print("="*40)
    
    def find_file(names):
        """Dosyayı ana dizinde veya 'resimler' klasöründe arar."""
        if isinstance(names, str): names = [names]
        for name in names:
            # 1. Ana dizine bak
            if os.path.exists(name): return name
            # 2. resimler klasörüne bak
            path_in_folder = os.path.join("resimler", name)
            if os.path.exists(path_in_folder): return path_in_folder
        return None

    # 1. Kapak Resmi Seçimi
    # Sırasıyla bu isimlere bakar
    cover_path = find_file(["kapak.jpg", "kapak.png", "manzara.jpg", "tablo.jpg"])
    
    if not cover_path:
        print("UYARI: Kapak resmi (kapak.jpg veya resimler/manzara.jpg) bulunamadı.")
        return

    # Kapak resmini yükle
    cover_img = prepare_single_image(cover_path)
    if cover_img is None: return
    cover_input = np.expand_dims(cover_img, axis=0)

    # 2. Test Senaryolarını Belirle
    scenarios = []
    
    # Senaryo A: Kullanıcı Resmi
    secret_img_path = find_file(["gizli.jpg", "gizli.png"])
    if secret_img_path:
        scenarios.append({
            "type": "image",
            "path": secret_img_path,
            "desc": f"Kullanici Resmi ({os.path.basename(secret_img_path)})",
            "save_name": "ozel_sonuc_resim.png"
        })
        
    # Senaryo B: Kullanıcı Metni
    secret_txt_path = find_file("gizli.txt")
    if secret_txt_path:
        scenarios.append({
            "type": "text",
            "path": secret_txt_path,
            "desc": f"Gizli Metin ({os.path.basename(secret_txt_path)})",
            "save_name": "ozel_sonuc_metin.png"
        })
        
    # Senaryo C: Hiçbiri yoksa Varsayılan
    if not scenarios:
        default_cat = find_file(["kedi.jpg", "resimler/kedi.jpg"])
        if default_cat:
            scenarios.append({
                "type": "image",
                "path": default_cat,
                "desc": "Varsayılan Kedi",
                "save_name": "ozel_sonuc.png"
            })
        else:
            print("Gizli saklanacak bir şey (gizli.jpg, gizli.txt veya kedi.jpg) bulunamadı!")
            return

    # 3. Her Senaryoyu Çalıştır
    for sc in scenarios:
        print(f"\n--- İşleniyor: {sc['desc']} ---")
        
        secret_img = None
        if sc["type"] == "image":
            secret_img = prepare_single_image(sc["path"])
        elif sc["type"] == "text":
            try:
                with open(sc["path"], 'r', encoding='utf-8') as f:
                    text = f.read()
                pil_img = text_to_image(text)
                secret_img = np.array(pil_img) / 255.0
            except Exception as e:
                print(f"Metin okuma hatası: {e}")
                continue
                
        if secret_img is None: continue
        
        secret_input = np.expand_dims(secret_img, axis=0)
        
        # Tahmin
        container_pred, reveal_pred = model.predict([cover_input, secret_input])
        
        # Görselleştirme
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        titles = [f"Kapak: {os.path.basename(cover_path)}", f"Gizli: {sc['desc']}", "Konteyner", "Geri Çıkarılan"]
        images_show = [cover_img, secret_img, container_pred[0], reveal_pred[0]]
        
        for i, ax in enumerate(axes):
            ax.imshow(images_show[i])
            ax.set_title(titles[i], fontsize=10)
            ax.axis('off')
            
        plt.savefig(sc["save_name"])
        plt.close() # Belleği temizle
        print(f"✅ Sonuç kaydedildi: {sc['save_name']}")

    print("-" * 50)
    print("TÜM İŞLEMLER TAMAMLANDI!")
    print("-" * 50)

# ==========================================
# 4. ANA ÇALIŞTIRMA BLOĞU
# ==========================================
if __name__ == "__main__":
    # 1. Veriyi Yükle
    # 'resimler' klasörüne bakacak, yoksa indirip oluşturacak
    covers, secrets = load_data()
    
    # 2. Modeli Oluştur
    print("Model inşa ediliyor...")
    stego_model = make_model()
    # stego_model.summary() # Detaylı tabloyu görmek istersen açabilirsin
    
    # 3. Modeli Derle (Compile)
    # İki çıktımız var: output_container ve output_reveal
    # İkisi için de Mean Squared Error (MSE) kullanıyoruz.
    # loss_weights ile beta katsayısını uyguluyoruz.
    stego_model.compile(
        optimizer='adam',
        loss={
            'output_container': 'mse', 
            'output_reveal': 'mse'
        },
        loss_weights={
            'output_container': 1.0,  # Kapak resminin bozulmaması ne kadar önemli?
            'output_reveal': BETA     # Gizli mesajın bulunması ne kadar önemli?
        }
    )
    
    # 4. Eğitimi Başlat
    print(f"Eğitim başlıyor ({EPOCHS} Epoch)... Lütfen bekleyin.")
    history = stego_model.fit(
        x=[covers, secrets],              # Girdiler
        y=[covers, secrets],              # Hedefler: Container -> Cover'a benzemeli, Reveal -> Secret'a benzemeli
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1
    )
    
    print("Eğitim tamamlandı!")
    
    # 5. Sonuçları Test Et ve Görselleştir
    print("Sonuçlar test ediliyor...")
    
    # Test için rastgele bir örnek seç
    idx = np.random.randint(0, len(covers))
    test_cover = covers[idx:idx+1]
    test_secret = secrets[idx:idx+1]
    
    # Tahmin yap
    container_pred, reveal_pred = stego_model.predict([test_cover, test_secret])
    
    # Görselleştirme
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Başlıklar
    titles = ["Orijinal Kapak (Cover)", "Gizli Resim (Secret)", "Konteyner (Container)", "Ortaya Çıkan (Revealed)"]
    images_show = [test_cover[0], test_secret[0], container_pred[0], reveal_pred[0]]
    
    for i, ax in enumerate(axes):
        ax.imshow(images_show[i])
        ax.set_title(titles[i])
        ax.axis('off')
    
    # Kaydet
    save_path = "sonuc.png"
    plt.savefig(save_path)
    print(f"Sonuç görseli '{save_path}' olarak kaydedildi. Dosyayı açıp bakabilirsin!")
    
    # Farkı hesapla (İsteğe bağlı bilgi)
    diff = np.abs(test_cover[0] - container_pred[0])
    print(f"Kapak ve Konteyner arasındaki ortalama piksel farkı: {np.mean(diff):.4f} (0-1 arası)")

    # 6. Özel Testi Çalıştır
    run_custom_test(stego_model)
