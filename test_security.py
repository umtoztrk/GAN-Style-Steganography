import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import sys
import numpy as np

# ==========================================
# 1. AYARLAR
# ==========================================
STEGO_CP_PATH = "latest.pt"           
PURE_CP_PATH  = "G_pure_pretrained.pt" 
IMAGE_PATH    = "test_image.jpg"      
IMG_SIZE      = 512
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🚀 Cihaz: {DEVICE}")

# ==========================================
# 2. MİMARİ YÜKLEME
# ==========================================
try:
    # SRNet'i de import ediyoruz
    from models import StegoGenerator, Extractor, PureGenerator, SRNet
    print("✅ Mimari (Dahil: SRNet) 'models.py' dosyasından yüklendi.")
except ImportError as e:
    print(f"❌ HATA: 'models.py' içinde eksik var! {e}")
    print("   Lütfen SRNet kodlarını models.py'a eklediğinden emin ol.")
    sys.exit(1)

# ==========================================
# 3. YARDIMCI FONKSİYONLAR
# ==========================================
def str_to_tensor(text, img_size):
    raw_bytes = text.encode('utf-8') + b'\x00' 
    bits = []
    for b in raw_bytes:
        for i in range(8):
            bits.append((b >> (7 - i)) & 1)
    bit_tensor = torch.tensor(bits, dtype=torch.float32)
    total_capacity = 3 * img_size * img_size
    msg_len = len(bit_tensor)
    if msg_len > total_capacity: return None, None
    padding_len = total_capacity - msg_len
    if padding_len > 0:
        noise = torch.randint(0, 2, (padding_len,)).float()
        full_tensor = torch.cat([bit_tensor, noise])
    else:
        full_tensor = bit_tensor
    return full_tensor.view(1, 3, img_size, img_size).to(DEVICE), msg_len

def tensor_to_str(tensor, msg_len):
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
    orig_flat = original.view(-1).cpu()
    rec_flat = (recovered.view(-1) > 0.5).float().cpu()
    orig_msg = orig_flat[:msg_len]
    rec_msg = rec_flat[:msg_len]
    acc_msg = (orig_msg == rec_msg).sum().item() / msg_len
    acc_total = (orig_flat == rec_flat).sum().item() / orig_flat.numel()
    return acc_msg, acc_total

def analyze_security(D, img_tensor, label_name):
    """Discriminator'a resmi sorar: Stego mu Clean mi?"""
    with torch.no_grad():
        logits = D(img_tensor)
        probs = F.softmax(logits, dim=1) # Olasılıklara çevir
        
        # Eğitimdeki etiketler: 0=Stego, 1=Clean
        prob_stego = probs[0, 0].item()
        prob_clean = probs[0, 1].item()
        
        pred_idx = torch.argmax(probs, dim=1).item()
        pred_label = "STEGO (Yakalandı!)" if pred_idx == 0 else "CLEAN (Kandırdık!)"
        
        return pred_label, prob_stego, prob_clean

# ==========================================
# 4. ANA TEST
# ==========================================
def run_security_test():
    print("\n" + "="*60)
    print("🕵️  GÜVENLİK & PERFORMANS TESTİ")
    print("="*60)

    # --- Modelleri Başlat ---
    G = StegoGenerator().to(DEVICE)
    E = Extractor().to(DEVICE)
    D = SRNet().to(DEVICE) # Discriminator
    
    # --- Yükleme ---
    if not os.path.exists(STEGO_CP_PATH):
        print(f"❌ Dosya yok: {STEGO_CP_PATH}")
        return

    try:
        ckpt = torch.load(STEGO_CP_PATH, map_location=DEVICE)
        
        # G ve E Yükle
        if 'G_state' in ckpt:
            G.load_state_dict(ckpt['G_state'])
            E.load_state_dict(ckpt['E_state'])
            # D Yükle (Eğer checkpointte varsa)
            if 'D_state' in ckpt:
                D.load_state_dict(ckpt['D_state'])
                print("✅ Discriminator (D) yüklendi.")
            else:
                print("⚠️ Uyarı: Checkpoint içinde 'D_state' bulunamadı! Rastgele D ile test anlamsız olur.")
            
            iter_num = ckpt.get('iteration', '?')
        else:
            print("❌ Bu checkpoint formatı D analizini desteklemiyor (Sadece G var).")
            return
            
        print(f"✅ Modeller Hazır (Iter: {iter_num})")
        G.eval(); E.eval(); D.eval()
    except Exception as e:
        print(f"❌ Hata: {e}")
        return

    # --- Resim ---
    if not os.path.exists(IMAGE_PATH): return
    img = Image.open(IMAGE_PATH).convert('RGB')
    transform = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()])
    cover = transform(img).unsqueeze(0).to(DEVICE)

    # --- Test ---
    text = input("\n✍️  Test Mesajı: ") or "Security Test 123"
    secret_tensor, msg_len = str_to_tensor(text, IMG_SIZE)
    if secret_tensor is None: return

    # --- İşlem ---
    with torch.no_grad():
        stego_img = G(cover, secret_tensor)
        recovered = E(stego_img)
        
        # Clean Model Varsa Karşılaştırma İçin Üret
        clean_ref = None
        if os.path.exists(PURE_CP_PATH):
            G_pure = PureGenerator().to(DEVICE)
            G_pure.load_state_dict(torch.load(PURE_CP_PATH, map_location=DEVICE))
            G_pure.eval()
            clean_ref = G_pure(cover)

    # --- Analizler ---
    msg_acc, _ = calculate_accuracy(secret_tensor, recovered, msg_len)
    
    # 1. Stego Resim Analizi
    d_res, p_stego, p_clean = analyze_security(D, stego_img, "Stego Image")
    
    # 2. Clean Resim Analizi (Varsa)
    if clean_ref is not None:
        d_res_c, p_stego_c, p_clean_c = analyze_security(D, clean_ref, "Clean Image")

    # --- RAPOR ---
    print("\n" + "-"*60)
    print("📊 GÜVENLİK RAPORU")
    print("-" * 60)
    print(f"📥 Mesaj Doğruluğu:   %{msg_acc*100:.2f}")
    print("-" * 60)
    
    print(f"🖼️  [STEGO RESİM] Analizi:")
    print(f"    D Tahmini:      {d_res}")
    print(f"    Stego Olasılığı: %{p_stego*100:.2f}")
    print(f"    Clean Olasılığı: %{p_clean*100:.2f}")
    
    if p_clean > p_stego:
        print("    ✅ BAŞARILI! (D bunun temiz resim olduğunu sanıyor)")
    else:
        print("    ❌ BAŞARISIZ! (D bunun sahte olduğunu anladı)")
        
    if clean_ref is not None:
        print("-" * 60)
        print(f"🖼️  [CLEAN RESİM] Analizi (Referans):")
        print(f"    D Tahmini:      {d_res_c}")
        print(f"    Clean Olasılığı: %{p_clean_c*100:.2f}")

    # Görsel kaydetme
    stego_np = stego_img.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
    Image.fromarray((stego_np * 255).clip(0, 255).astype(np.uint8)).save("security_test_stego.png")
    print("\n💾 'security_test_stego.png' kaydedildi.")

if __name__ == "__main__":
    run_security_test()