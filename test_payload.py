import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import time
import numpy as np
import sys

# ==========================================
# 1. AYARLAR
# ==========================================
STEGO_CP_PATH = "latest.pt"           # Stego Model
PURE_CP_PATH  = "G_pure_pretrained.pt" # Clean (Pure) Model
IMAGE_PATH    = "test_image.jpg"      
IMG_SIZE      = 512
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🚀 Cihaz: {DEVICE}")

# ==========================================
# 2. MİMARİ YÜKLEME (models.py'dan)
# ==========================================
try:
    # PureGenerator'ı da ekledik
    from models import StegoGenerator, Extractor, PureGenerator
    print("✅ Mimari 'models.py' dosyasından yüklendi.")
except ImportError:
    print("❌ HATA: 'models.py' bulunamadı!")
    sys.exit(1)

# ==========================================
# 3. YARDIMCI FONKSİYONLAR
# ==========================================
def str_to_tensor(text, img_size):
    """Metni ham bitlere (tensor) çevirir."""
    raw_bytes = text.encode('utf-8') + b'\x00' 
    bits = []
    for b in raw_bytes:
        for i in range(8):
            bits.append((b >> (7 - i)) & 1)
            
    bit_tensor = torch.tensor(bits, dtype=torch.float32)
    total_capacity = 3 * img_size * img_size
    msg_len = len(bit_tensor)
    
    if msg_len > total_capacity:
        print(f"❌ Metin çok uzun! ({msg_len} > {total_capacity})")
        return None, None
    
    padding_len = total_capacity - msg_len
    if padding_len > 0:
        noise = torch.randint(0, 2, (padding_len,)).float()
        full_tensor = torch.cat([bit_tensor, noise])
    else:
        full_tensor = bit_tensor
        
    return full_tensor.view(1, 3, img_size, img_size).to(DEVICE), msg_len

def tensor_to_str(tensor, msg_len):
    """Model çıktısını metne çevirir."""
    preds = (tensor.view(-1) > 0.5).int().cpu().numpy()
    msg_bits = preds[:msg_len]
    byte_data = bytearray()
    for i in range(0, len(msg_bits), 8):
        chunk = msg_bits[i:i+8]
        if len(chunk) < 8: break
        val = 0
        for bit in chunk:
            val = (val << 1) | int(bit)
        byte_data.append(val)
    return byte_data.decode('utf-8', errors='replace')

def calculate_accuracy(original, recovered, msg_len):
    """Bit bazında doğruluk oranını hesaplar."""
    orig_flat = original.view(-1).cpu()
    rec_flat = (recovered.view(-1) > 0.5).float().cpu()
    
    # 1. Payload Accuracy
    orig_msg = orig_flat[:msg_len]
    rec_msg = rec_flat[:msg_len]
    correct_msg = (orig_msg == rec_msg).sum().item()
    acc_msg = correct_msg / msg_len
    
    # 2. Full Image Accuracy
    correct_total = (orig_flat == rec_flat).sum().item()
    acc_total = correct_total / orig_flat.numel()
    
    return acc_msg, acc_total

def save_image(tensor, name):
    """Tensörü resim olarak kaydeder."""
    img_np = tensor.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(img_np).save(name)
    print(f"💾 Kaydedildi: {name}")

# ==========================================
# 4. ANA TEST FONKSİYONU
# ==========================================
def run_accuracy_test():
    print("\n" + "="*60)
    print("📊 HAM MODEL PERFORMANS TESTİ (STEGO + CLEAN)")
    print("="*60)

    # --- Modelleri Başlat ---
    G = StegoGenerator().to(DEVICE)
    E = Extractor().to(DEVICE)
    G_pure = PureGenerator().to(DEVICE) # Clean Model
    
    # --- Yükleme: Stego ---
    if not os.path.exists(STEGO_CP_PATH):
        print(f"❌ Stego Model dosyası yok: {STEGO_CP_PATH}")
        return

    try:
        ckpt = torch.load(STEGO_CP_PATH, map_location=DEVICE)
        if 'G_state' in ckpt:
            G.load_state_dict(ckpt['G_state'])
            E.load_state_dict(ckpt['E_state'])
            iter_num = ckpt.get('iteration', '?')
        else:
            G.load_state_dict(ckpt)
            iter_num = "Bilinmiyor"
        print(f"✅ Stego Model Yüklendi (Iter: {iter_num})")
    except Exception as e:
        print(f"❌ Stego Model Hatası: {e}")
        return

    # --- Yükleme: Clean (Pure) ---
    if os.path.exists(PURE_CP_PATH):
        try:
            # Genelde Pure model sadece state_dict olarak kaydedilir
            G_pure.load_state_dict(torch.load(PURE_CP_PATH, map_location=DEVICE))
            print(f"✅ Pure Model Yüklendi: {PURE_CP_PATH}")
            has_pure = True
        except Exception as e:
            print(f"⚠️ Pure Model yüklenirken hata: {e}")
            has_pure = False
    else:
        print(f"⚠️ Pure Model dosyası bulunamadı ({PURE_CP_PATH}). Clean resim üretilmeyecek.")
        has_pure = False

    G.eval(); E.eval(); G_pure.eval()

    # --- Resim Hazırlığı ---
    if not os.path.exists(IMAGE_PATH):
        print("❌ Resim yok!")
        return
    img = Image.open(IMAGE_PATH).convert('RGB')
    transform = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()])
    cover = transform(img).unsqueeze(0).to(DEVICE)

    # --- Test ---
    text = input("\n✍️  Test Mesajı Girin: ") or "Derin Ogrenme ve Steganografi Testi 2024"
    print(f"\n⚙️  İşleniyor: '{text}'")
    
    # 1. Kodlama
    secret_tensor, msg_bit_len = str_to_tensor(text, IMG_SIZE)
    if secret_tensor is None: return

    # 2. Modelden Geçiş
    with torch.no_grad():
        # Stego Üretimi
        stego_img = G(cover, secret_tensor)
        
        # Clean Üretimi (Eğer model varsa)
        clean_img = None
        if has_pure:
            clean_img = G_pure(cover)
            
        # Mesaj Çıkarma
        recovered_tensor = E(stego_img)

    # 3. Analiz
    msg_acc, total_acc = calculate_accuracy(secret_tensor, recovered_tensor, msg_bit_len)
    decoded_text = tensor_to_str(recovered_tensor, msg_bit_len)

    # --- Raporlama ---
    print("\n" + "-"*60)
    print("📈 SONUÇ RAPORU")
    print("-" * 60)
    print(f"📥 Giden Mesaj:     {text}")
    print(f"📤 Çıkan Mesaj:     {decoded_text}")
    print("-" * 60)
    print(f"🎯 Mesaj Doğruluğu:  %{msg_acc*100:.2f} (Sadece metin bitleri)")
    print(f"🌍 Genel Doğruluk:   %{total_acc*100:.2f} (Tüm 512x512 alan)")
    print("-" * 60)
    
    # Yorum
    if msg_acc == 1.0: print("🏆 MÜKEMMEL! (%100)")
    elif msg_acc > 0.95: print("✅ ÇOK İYİ (%95+).")
    elif msg_acc > 0.85: print("⚠️ ORTA SEVİYE.")
    else: print("❌ KÖTÜ.")

    # --- Kaydetme ---
    print("\n💾 DOSYALAR KAYDEDİLİYOR...")
    save_image(stego_img, "test_stego_result.png")
    
    if clean_img is not None:
        save_image(clean_img, "test_clean_result.png")
    else:
        print("ℹ️  Clean model olmadığı için 'test_clean_result.png' oluşturulmadı.")

if __name__ == "__main__":
    run_accuracy_test()