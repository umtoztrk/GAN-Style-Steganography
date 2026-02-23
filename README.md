# 🎨 GAN-Style-Steganography: Coverless Data Hiding via Style Transfer

> An unofficial, enhanced PyTorch implementation of the paper *"Image Steganography and Style Transformation Based on Generative Adversarial Network" (Li et al., 2024)*. 

This project introduces a "Coverless" steganography paradigm by embedding secret messages directly during the neural style transfer process using a Generative Adversarial Network (GAN). By eliminating the original "natural image" baseline, it provides **Plausible Deniability** against modern steganalysis. 

Furthermore, this repository enhances the original paper's architecture by introducing a **Channel Coding (Repetition Code) mechanism**, achieving a **100% message recovery rate** in practical text transmissions.

---

## ✨ Key Features

* **Coverless Steganography:** The model does not modify an existing cover image. Instead, it generates a stylized stego image from scratch, making comparison-based steganalysis highly difficult.
* **4-Part GAN Architecture:** Integrates a Generator (Message-Embedding), an Extractor (Message-Extraction), a Discriminator (SRNet Steganalyzer), and a pre-trained VGG-19 Loss Network.
* **SRNet Discriminator:** Uses the state-of-the-art SRNet architecture as an adversarial discriminator to ensure the generated stego images are statistically indistinguishable from clean style-transferred images.
* **Error Correction via Channel Coding:** Implements Majority Voting to correct pixel-level interpretation errors, boosting raw bit accuracy from ~92.5% to a perfect 100% payload recovery.

---

## 🏗️ Architecture & Methodology


The framework optimizes a complex objective function balancing artistic style, content preservation, message recovery, and security against the steganalyzer:

$$L_{total} = \alpha L_{cont} + \beta L_{sty} + \lambda L_{ext} - \gamma L_{adv}$$

* **$L_{cont}$ (Content Loss):** Ensures the output maintains the structural elements of the original photo.
* **$L_{sty}$ (Style Loss):** Ensures the output adopts the desired artistic texture (calculated via Gram Matrices).
* **$L_{ext}$ (Extraction Loss):** Guarantees accurate recovery of the hidden data by the receiver.
* **$L_{adv}$ (Adversarial Loss):** Evaluates the Discriminator's ability to classify the output; maximized for the Generator to confuse the SRNet.

---

## 📂 Repository Structure

```text
📁 GAN-Style-Steganography
│
├── 📄 models.py                # Core Neural Network Architectures (StegoGen, Extractor, SRNet, PureGen)
├── 📄 test_payload.py          # Script for basic message embedding, extraction, and PSNR evaluation
├── 📄 test_security.py         # Script testing the stego image against the SRNet Discriminator
├── 📄 test_channel_coding.py   # Script demonstrating 100% message recovery via Majority Voting
│
├── 📄 latest.pt                # Pre-trained weights for the Stego Model (Generator & Extractor)
├── 📄 G_pure_pretrained.pt     # Pre-trained weights for the Clean Model (Baseline without steganography)
├── 🖼️ test_image.jpg           # Sample content image for quick testing
│
└── 📄 README.md                # Project documentation
```
## 🚀 Quick Start & Usage
1. Requirements
    Python 3.8+

    PyTorch & Torchvision

    Pillow (PIL)

    NumPy

2. Running the Tests

A. Basic Payload & Visual Quality Test
Evaluates the core embedding and extraction mechanics.
```bash
python test_payload.py
```
B. Steganalysis Security Test
Pits the generated stego image against the SRNet Discriminator.
```bash
python test_security.py
```
C. Channel Coding Test
Demonstrates how the repetition coding (x7) corrects physical layer errors to recover the exact text payload perfectly.
```bash
python test_channel_coding.py
```
## 📊 Experimental Results
Based on our testing and evaluation phase:

Visual Fidelity: Achieved a PSNR of 21.5 dB, rendering the steganographic modifications visually coherent with standard artistic transfers.

Raw Bit Accuracy: The base model achieves ~92.5% physical layer accuracy for text bits.

Enhanced Recovery: With Channel Coding enabled, the system successfully corrects the remaining ~7.5% error margin, yielding 100% Message Recovery.

Security Rating: The SRNet discriminator detection rate hovers around 60%, closely approaching the ideal Nash Equilibrium of 50% (random guessing).

##🔮 Future Work
While the core architecture successfully demonstrates coverless steganography, we plan to improve the system in the following areas:
Graphical User Interface (GUI): Transitioning from command-line scripts to a user-friendly application interface to make the system accessible to non-technical end-users.
Perfecting the Nash Equilibrium: Currently, the discriminator detection rate is ~60%. We aim to train the model longer with a more aggressive adversarial weight ($\gamma$) to push this down to exactly 50% (random guessing).
Hyper-Realistic Visual Quality: Refining hyperparameters and warmup strategies to push the image visual fidelity (PSNR) from 21.5 dB to >30 dB.
Bridging the Raw Accuracy Gap: Improving the base model's raw bit extraction reliability closer to 99% before applying any channel coding.

## 🎓 Credits & References
Developers: 
- Umut Öztürk 
- Eren Eroğlu

Base Paper: Li, L., Zhang, X., Chen, K., Feng, G., Wu, D., & Zhang, W. (2024). Image Steganography and Style Transformation Based on Generative Adversarial Network. Mathematics, 12(4), 615.


