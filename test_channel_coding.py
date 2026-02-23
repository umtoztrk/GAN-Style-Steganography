import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import sys
import numpy as np

# ==========================================
# 1. SETTINGS
# ==========================================
STEGO_CP_PATH = "latest.pt"           # Stego Model
PURE_CP_PATH  = "G_pure_pretrained.pt" # Clean (Pure) Model
IMAGE_PATH    = "test_image.jpg"      
IMG_SIZE      = 512
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REPEAT_N      = 7  # Channel Coding Factor (Must be an odd number: 3, 5, 7, etc.)

print(f"🚀 Device: {DEVICE}")
print(f"🛡️ Channel Coding: Repetition (x{REPEAT_N})")

# ==========================================
# 2. LOAD ARCHITECTURE (from models.py)
# ==========================================
try:
    from models import StegoGenerator, Extractor, PureGenerator
    print("✅ Architecture loaded from 'models.py'.")
except ImportError:
    print("❌ ERROR: 'models.py' not found!")
    sys.exit(1)

# ==========================================
# 3. HELPER FUNCTIONS (Channel Coding)
# ==========================================
def str_to_tensor_encoded(text, img_size):
    """
    Converts text to bits AND applies Channel Coding (Repetition).
    """
    # 1. Text -> Bytes -> Bits
    raw_bytes = text.encode('utf-8') + b'\x00' 
    raw_bits = []
    for b in raw_bytes:
        for i in range(8):
            raw_bits.append((b >> (7 - i)) & 1)
            
    # 2. APPLY CHANNEL CODING (Repetition)
    # 1 -> 1111111
    encoded_bits = []
    for b in raw_bits:
        encoded_bits.extend([b] * REPEAT_N)
        
    bit_tensor = torch.tensor(encoded_bits, dtype=torch.float32)
    
    # 3. Capacity Check
    total_capacity = 3 * img_size * img_size
    encoded_len = len(bit_tensor)
    
    if encoded_len > total_capacity:
        print(f"❌ Text too long! ({encoded_len} > {total_capacity})")
        print(f"   (Due to x{REPEAT_N} repetition, max capacity is reduced.)")
        return None, None, None
    
    # 4. Padding
    padding_len = total_capacity - encoded_len
    if padding_len > 0:
        noise = torch.randint(0, 2, (padding_len,)).float()
        full_tensor = torch.cat([bit_tensor, noise])
    else:
        full_tensor = bit_tensor
        
    return full_tensor.view(1, 3, img_size, img_size).to(DEVICE), encoded_len, len(raw_bits)

def tensor_to_str_decoded(tensor, encoded_len, original_bit_len):
    """
    Decodes the model output using Majority Voting.
    """
    # Thresholding
    preds = (tensor.view(-1) > 0.5).int().cpu().tolist()
    
    # Isolate the relevant message part
    encoded_msg_bits = preds[:encoded_len]
    
    # --- DECODING (Majority Voting) ---
    recovered_bits = []
    
    # Read in chunks of REPEAT_N
    for i in range(0, len(encoded_msg_bits), REPEAT_N):
        chunk = encoded_msg_bits[i:i+REPEAT_N]
        if len(chunk) < REPEAT_N: break
        
        # Voting: If more than half are 1, then 1. Else 0.
        vote = 1 if sum(chunk) > (REPEAT_N / 2) else 0
        recovered_bits.append(vote)
        
    # Bits -> Bytes -> String
    byte_data = bytearray()
    for i in range(0, len(recovered_bits), 8):
        chunk = recovered_bits[i:i+8]
        if len(chunk) < 8: break
        val = 0
        for bit in chunk:
            val = (val << 1) | int(bit)
        byte_data.append(val)
        
    # --- DÜZELTME BURADA ---
    # NULL (\x00) karakterini görünce kesip atıyoruz
    decoded_str = byte_data.decode('utf-8', errors='replace')
    if '\x00' in decoded_str:
        decoded_str = decoded_str.split('\x00')[0]
        
    return decoded_str

def calculate_accuracy(original_encoded, recovered_tensor, encoded_len):
    """Calculates Raw Bit Accuracy (Before Correction)."""
    orig_flat = original_encoded.view(-1).cpu()
    rec_flat = (recovered_tensor.view(-1) > 0.5).float().cpu()
    
    orig_msg = orig_flat[:encoded_len]
    rec_msg = rec_flat[:encoded_len]
    
    correct_msg = (orig_msg == rec_msg).sum().item()
    acc_msg = correct_msg / encoded_len
    
    return acc_msg

def save_image(tensor, name):
    img_np = tensor.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(img_np).save(name)
    print(f"💾 Saved: {name}")

# ==========================================
# 4. MAIN TEST FUNCTION
# ==========================================
def run_channel_coding_test():
    print("\n" + "="*60)
    print(f"📊 PERFORMANCE TEST WITH CHANNEL CODING (x{REPEAT_N})")
    print("="*60)

    # --- Initialize Models ---
    G = StegoGenerator().to(DEVICE)
    E = Extractor().to(DEVICE)
    G_pure = PureGenerator().to(DEVICE)
    
    # --- Load Stego ---
    if not os.path.exists(STEGO_CP_PATH):
        print(f"❌ Stego Model not found: {STEGO_CP_PATH}")
        return

    try:
        ckpt = torch.load(STEGO_CP_PATH, map_location=DEVICE)
        if 'G_state' in ckpt:
            G.load_state_dict(ckpt['G_state'])
            E.load_state_dict(ckpt['E_state'])
            iter_num = ckpt.get('iteration', '?')
        else:
            G.load_state_dict(ckpt)
            iter_num = "Unknown"
        print(f"✅ Stego Model Loaded (Iter: {iter_num})")
    except Exception as e:
        print(f"❌ Stego Model Error: {e}")
        return

    # --- Load Clean ---
    has_pure = False
    if os.path.exists(PURE_CP_PATH):
        try:
            G_pure.load_state_dict(torch.load(PURE_CP_PATH, map_location=DEVICE))
            print(f"✅ Pure Model Loaded: {PURE_CP_PATH}")
            has_pure = True
        except Exception as e:
            print(f"⚠️ Pure Model Error: {e}")

    G.eval(); E.eval(); G_pure.eval()

    # --- Prepare Image ---
    if not os.path.exists(IMAGE_PATH):
        print("❌ Image not found!")
        return
    img = Image.open(IMAGE_PATH).convert('RGB')
    transform = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()])
    cover = transform(img).unsqueeze(0).to(DEVICE)

    # --- Test Input ---
    text = input("\n✍️  Enter Test Message: ") or "Deep Learning & Channel Coding Test 2025"
    print(f"\n⚙️  Processing: '{text}'")
    
    # 1. Encode with Repetition
    # returns: tensor, length of encoded bits, length of original bits
    secret_tensor, encoded_len, raw_len = str_to_tensor_encoded(text, IMG_SIZE)
    if secret_tensor is None: return

    # 2. Model Forward Pass
    with torch.no_grad():
        stego_img = G(cover, secret_tensor)
        
        clean_img = None
        if has_pure:
            clean_img = G_pure(cover)
            
        recovered_tensor = E(stego_img)

    # 3. Analysis & Decoding
    # Calculate Raw Accuracy (Physical Layer)
    raw_acc = calculate_accuracy(secret_tensor, recovered_tensor, encoded_len)
    
    # Decode with Majority Voting (Application Layer)
    final_text = tensor_to_str_decoded(recovered_tensor, encoded_len, raw_len)

    # --- REPORT ---
    print("\n" + "-"*60)
    print("📈 FINAL REPORT")
    print("-" * 60)
    print(f"📥 Input Message:      {text}")
    print(f"📤 Recovered Message:  {final_text}")
    print("-" * 60)
    print(f"📡 Raw Bit Accuracy:   %{raw_acc*100:.2f} (Model Performance)")
    print(f"🛡️ Coding Strategy:    Repetition Code (x{REPEAT_N})")
    
    if text == final_text:
        print("\n🏆 RESULT: SUCCESS (100% Message Recovery)")
        if raw_acc < 1.0:
            print(f"   (Channel Coding corrected the {(1-raw_acc)*100:.2f}% error!)")
    else:
        print("\n❌ RESULT: FAILURE (Too much noise for coding to fix)")

    print("-" * 60)

    # --- Save Images ---
    print("\n💾 SAVING IMAGES...")
    save_image(stego_img, "test_stego_coded.png")
    if clean_img is not None:
        save_image(clean_img, "test_clean_result.png")

if __name__ == "__main__":
    run_channel_coding_test()