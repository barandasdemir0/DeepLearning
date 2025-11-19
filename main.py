"""
Deep Steganography Project
==========================
Bu modül, Derin Öğrenme (Deep Learning) kullanarak bir görüntüyü başka bir görüntünün içine
piksel seviyesinde gizleyen ve geri çıkaran profesyonel bir uygulama sunar.

Mimari:
    - Prep Network: Gizli veriyi hazırlar.
    - Hiding Network (Encoder): Veriyi kapak resmine gizler.
    - Reveal Network (Decoder): Gizli veriyi geri çıkarır.

Yazar: DeepMind Assistant
Tarih: 2025
"""

import os
import glob
from typing import Tuple, List, Optional, Union

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import requests
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

# ==========================================
# KONFİGÜRASYON (CONFIGURATION)
# ==========================================
class Config:
    """Proje genelindeki sabitler ve ayarlar."""
    IMG_SIZE = (128, 128)
    CHANNELS = 3
    BATCH_SIZE = 8
    EPOCHS = 100
    LEARNING_RATE = 0.001
    
    # Loss Ağırlıkları (GÜNCELLENDİ: Denge Ayarı)
    # Beta: Gizli resim kalitesi için artırıldı (Geri çıkarılan daha net olacak)
    # Alpha: Kapak koruması biraz gevşetildi (Çok hafif ghosting olabilir)
    LOSS_ALPHA = 5.0  
    LOSS_BETA = 1.5   
    
    DATA_DIR = "resimler"
    DEFAULT_URLS = [
        ("manzara.jpg", "https://images.unsplash.com/photo-1472214103451-9374bd1c798e?q=80&w=400&auto=format&fit=crop"),
        ("tablo.jpg", "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg/402px-Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg"),
        ("sehir.jpg", "https://images.unsplash.com/photo-1519501025264-65ba15a82390?q=80&w=400&auto=format&fit=crop"),
        ("kedi.jpg", "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?q=80&w=400&auto=format&fit=crop")
    ]

# ==========================================
# VERİ YÖNETİCİSİ (DATA MANAGER)
# ==========================================
class DataManager:
    """Veri indirme, yükleme ve işleme süreçlerini yönetir."""
    
    def __init__(self, data_dir: str = Config.DATA_DIR):
        self.data_dir = data_dir
        self._ensure_data_dir()
        
    def _ensure_data_dir(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
    def download_defaults(self):
        """Varsayılan resimleri indirir."""
        headers = {'User-Agent': 'Mozilla/5.0'}
        print(f"[INFO] Veri klasörü kontrol ediliyor: {self.data_dir}")
        
        for filename, url in Config.DEFAULT_URLS:
            path = os.path.join(self.data_dir, filename)
            if not os.path.exists(path):
                try:
                    print(f"   İndiriliyor: {filename}...")
                    response = requests.get(url, headers=headers, timeout=10)
                    if response.status_code == 200:
                        with open(path, 'wb') as f:
                            f.write(response.content)
                except Exception as e:
                    print(f"   [HATA] İndirme başarısız ({filename}): {e}")

    def load_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Eğitim için veri setini hazırlar."""
        self.download_defaults()
        
        valid_exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
        files = []
        for ext in valid_exts:
            files.extend(glob.glob(os.path.join(self.data_dir, ext)))
            
        if not files:
            print("[UYARI] Hiç resim bulunamadı! Rastgele gürültü kullanılacak.")
            return self._generate_noise_data()
            
        images = []
        for f in files:
            try:
                img = Image.open(f).convert('RGB')
                img = img.resize(Config.IMG_SIZE)
                images.append(np.array(img) / 255.0)
            except Exception as e:
                print(f"   [HATA] Resim okunamadı {f}: {e}")
                
        if not images:
            return self._generate_noise_data()
            
        # Data Augmentation (Veri Çoğaltma)
        X = np.array(images * 50) # Veri setini yapay olarak büyüt
        np.random.shuffle(X)
        
        # Split (Yarı yarıya böl)
        mid = len(X) // 2
        covers = X[:mid]
        secrets = X[mid:2*mid]
        
        # Boyut eşitleme
        limit = min(len(covers), len(secrets))
        return covers[:limit], secrets[:limit]

    def _generate_noise_data(self):
        shape = (10, *Config.IMG_SIZE, Config.CHANNELS)
        return np.random.rand(*shape), np.random.rand(*shape)

    @staticmethod
    def load_single_image(path: str) -> Optional[np.ndarray]:
        """Tek bir resmi yükler ve normalize eder."""
        try:
            img = Image.open(path).convert('RGB')
            img = img.resize(Config.IMG_SIZE)
            return np.array(img) / 255.0
        except:
            return None

    @staticmethod
    def text_to_image(text: str) -> np.ndarray:
        """Metni resme dönüştürür."""
        img = Image.new('RGB', Config.IMG_SIZE, color=(0, 0, 0))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()
            
        # Metni ortalamaya çalışmadan basitçe yaz
        draw.text((10, 50), text, fill=(255, 255, 255), font=font)
        return np.array(img) / 255.0

# ==========================================
# MODEL MİMARİSİ (STEGANOGRAPHY NETWORK)
# ==========================================
class StegoModel:
    """Deep Steganography Model Mimarisini Kapsüller."""
    
    def __init__(self):
        self.model = self._build_graph()
        
    def _build_graph(self) -> models.Model:
        # Girdiler
        input_cover = layers.Input(shape=(*Config.IMG_SIZE, Config.CHANNELS), name='cover_input')
        input_secret = layers.Input(shape=(*Config.IMG_SIZE, Config.CHANNELS), name='secret_input')
        
        # --- 1. Prep Network ---
        # Gizli resmin özelliklerini çıkarır
        p = layers.Conv2D(50, (3, 3), padding='same', activation='relu')(input_secret)
        p = layers.Conv2D(50, (3, 3), padding='same', activation='relu')(p)
        p = layers.Conv2D(50, (3, 3), padding='same', activation='relu')(p)
        p = layers.Conv2D(50, (3, 3), padding='same', activation='relu')(p)
        
        # --- 2. Hiding Network ---
        # Kapak ve Prep çıktısını birleştirir
        concat = layers.Concatenate()([input_cover, p])
        
        h = layers.Conv2D(50, (3, 3), padding='same', activation='relu')(concat)
        h = layers.Conv2D(50, (3, 3), padding='same', activation='relu')(h)
        h = layers.Conv2D(50, (3, 3), padding='same', activation='relu')(h)
        h = layers.Conv2D(50, (3, 3), padding='same', activation='relu')(h)
        h = layers.Conv2D(50, (3, 3), padding='same', activation='relu')(h)
        
        # Çıktı Katmanı (Container)
        output_container = layers.Conv2D(3, (3, 3), padding='same', activation='sigmoid', name='output_container')(h)
        
        # --- 3. Reveal Network ---
        # Container'dan gizli resmi çıkarır
        r = layers.Conv2D(50, (3, 3), padding='same', activation='relu')(output_container)
        r = layers.Conv2D(50, (3, 3), padding='same', activation='relu')(r)
        r = layers.Conv2D(50, (3, 3), padding='same', activation='relu')(r)
        r = layers.Conv2D(50, (3, 3), padding='same', activation='relu')(r)
        r = layers.Conv2D(50, (3, 3), padding='same', activation='relu')(r)
        
        output_reveal = layers.Conv2D(3, (3, 3), padding='same', activation='sigmoid', name='output_reveal')(r)
        
        return models.Model(inputs=[input_cover, input_secret], outputs=[output_container, output_reveal])
    
    def compile(self):
        """Modeli özel ağırlıklarla derler."""
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE),
            loss={
                'output_container': 'mse',
                'output_reveal': 'mse'
            },
            loss_weights={
                'output_container': Config.LOSS_ALPHA, # Kapak resminin bozulmaması ÇOK önemli
                'output_reveal': Config.LOSS_BETA      # Gizli resmin çıkması da önemli
            }
        )
        
    def train(self, covers, secrets):
        """Eğitim döngüsünü başlatır."""
        print(f"\n[INFO] Eğitim Başlıyor ({Config.EPOCHS} Epoch)...")
        print(f"       Alpha (Kapak Koruması): {Config.LOSS_ALPHA}")
        print(f"       Beta  (Gizli Veri):     {Config.LOSS_BETA}")
        
        history = self.model.fit(
            x=[covers, secrets],
            y=[covers, secrets],
            batch_size=Config.BATCH_SIZE,
            epochs=Config.EPOCHS,
            verbose=1
        )
        return history

    def predict(self, cover, secret):
        """Tekil tahmin yapar."""
        # Boyut ekle (Batch dimension): (128,128,3) -> (1,128,128,3)
        c_in = np.expand_dims(cover, axis=0)
        s_in = np.expand_dims(secret, axis=0)
        return self.model.predict([c_in, s_in], verbose=0)

# ==========================================
# GÖRSELLEŞTİRİCİ (VISUALIZER)
# ==========================================
class Visualizer:
    """Sonuçları görselleştirme ve kaydetme işlemlerini yapar."""
    
    @staticmethod
    def save_results(cover, secret, container, revealed, filename="sonuc.png", title_suffix=""):
        """4'lü sonuç görselini kaydeder."""
        fig, axes = plt.subplots(1, 4, figsize=(16, 5))
        
        items = [
            ("Orijinal Kapak", cover),
            (f"Gizli Veri\n{title_suffix}", secret),
            ("Stego (Konteyner)", container),
            ("Geri Çıkarılan", revealed)
        ]
        
        for i, (title, img) in enumerate(items):
            # Görüntüyü kırp (0-1 dışına taşanları düzelt)
            img = np.clip(img, 0, 1)
            axes[i].imshow(img)
            axes[i].set_title(title, fontsize=10, fontweight='bold')
            axes[i].axis('off')
            
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"   [KAYIT] Sonuç kaydedildi: {filename}")

# ==========================================
# ANA UYGULAMA (MAIN APP)
# ==========================================
def find_local_file(names: Union[str, List[str]]) -> Optional[str]:
    """Dosyayı kök dizinde veya data klasöründe arar."""
    if isinstance(names, str): names = [names]
    for name in names:
        if os.path.exists(name): return name
        path_in_data = os.path.join(Config.DATA_DIR, name)
        if os.path.exists(path_in_data): return path_in_data
    return None

def run_custom_tests(model: StegoModel):
    """Kullanıcının özel dosyalarıyla test yapar."""
    print("\n" + "="*50)
    print("      ÖZEL TEST SENARYOLARI ÇALIŞTIRILIYOR")
    print("="*50)
    
    # 1. Kapak Resmi Bul
    cover_path = find_local_file(["kapak.jpg", "kapak.png", "manzara.jpg"])
    if not cover_path:
        print("[UYARI] Kapak resmi bulunamadı!")
        return
        
    cover_img = DataManager.load_single_image(cover_path)
    if cover_img is None: return

    # 2. Senaryoları Belirle
    scenarios = []
    
    # A. Resim Saklama
    secret_img_path = find_local_file(["gizli.jpg", "gizli.png"])
    if secret_img_path:
        scenarios.append({
            "type": "img", "data": secret_img_path, 
            "name": "ozel_sonuc_resim.png", "desc": "Resim Saklama"
        })
        
    # B. Metin Saklama
    secret_txt_path = find_local_file("gizli.txt")
    if secret_txt_path:
        scenarios.append({
            "type": "txt", "data": secret_txt_path, 
            "name": "ozel_sonuc_metin.png", "desc": "Metin Saklama"
        })
        
    # C. Varsayılan (Eğer hiçbiri yoksa)
    if not scenarios:
        cat_path = find_local_file(["kedi.jpg", "resimler/kedi.jpg"])
        if cat_path:
            scenarios.append({
                "type": "img", "data": cat_path, 
                "name": "ozel_sonuc_demo.png", "desc": "Demo (Kedi)"
            })

    # 3. Çalıştır
    for sc in scenarios:
        print(f"\n--- Senaryo: {sc['desc']} ---")
        
        secret_img = None
        if sc['type'] == 'img':
            secret_img = DataManager.load_single_image(sc['data'])
        elif sc['type'] == 'txt':
            try:
                with open(sc['data'], 'r', encoding='utf-8') as f:
                    text = f.read()
                secret_img = DataManager.text_to_image(text)
            except:
                print("   [HATA] Metin dosyası okunamadı.")
                continue
                
        if secret_img is None: continue
        
        # Tahmin
        preds = model.predict(cover_img, secret_img)
        container, revealed = preds[0][0], preds[1][0]
        
        # Kaydet
        Visualizer.save_results(
            cover_img, secret_img, container, revealed, 
            filename=sc['name'], title_suffix=f"({sc['desc']})"
        )

def main():
    # 1. Veri Hazırlığı
    dm = DataManager()
    covers, secrets = dm.load_dataset()
    print(f"[INFO] Veri Seti: {len(covers)} çift resim hazırlandı.")
    
    # 2. Model Kurulumu
    stego = StegoModel()
    stego.compile()
    # stego.model.summary() # İstersen açabilirsin
    
    # 3. Eğitim
    stego.train(covers, secrets)
    
    # 4. Testler
    run_custom_tests(stego)
    
    print("\n[BAŞARILI] Tüm işlemler tamamlandı.")

if __name__ == "__main__":
    main()
