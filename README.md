# GameForge AI 🎮🤖

Welcome to **GameForge AI**! This project is a multi-model Generative AI pipeline designed for 2D game asset design, semantic planning, adversarial sprite synthesis, latent space analysis, and high-fidelity reconstruction.

---

> **Fork note:** This is a team project built with [Manish-korde](https://github.com/Manish-korde) — original repository: [Manish-korde/GameForge-AI](https://github.com/Manish-korde/GameForge-AI). This fork highlights my contributions.

## 👤 My Contributions ([Swayamvijay43](https://github.com/Swayamvijay43))

### 1. Fine-tuned Transformer — natural language → structured JSON game spec
* Fine-tuned `google/flan-t5-small` (HuggingFace Transformers, PyTorch) to turn plain-English game ideas into a JSON spec (`theme`, `environment`, `visualStyle`, `characters[]`, `weapons[]`, `props[]`).
* Built the dataset: **2,000 unique prompt → JSON pairs** (1,600 / 200 / 200 split, 30+ themes).
* Found that T5's tokenizer has no `{` / `}` tokens, added custom brace tokens and retrained.
* Results on 200 held-out examples: **100% valid JSON**, **100% schema match**, **97.5% theme accuracy**.
* Integrated into the FastAPI backend.

### 2. Neural pose deformation for sprite animation (`feature/gameforge-pose-deformation` branch)
* Turns one static sprite into `walk_1`, `walk_2`, `run`, `jump` and `attack_slash` frames.
* Geometric baseline: 16-joint 2D skeleton with Linear Blend Skinning and nearest-neighbour resampling to keep pixel art crisp.
* Neural model: `TorchFlowSTN`, a PyTorch spatial transformer network that predicts a dense flow field and warps the sprite with `F.grid_sample`.
* Debugged training collapse via gradient tracing (LeakyReLU, zero-initialised flow head, zero padding, alpha snapping).
* Results on 35 held-out samples: **L1 error −69%**, **PSNR +6.4 dB**, **silhouette IoU 0.957**, **100% of blurry outline pixels removed**, **~236 FPS** on an RTX 4060.
* Limitation: training targets come from the geometric rig, so the network learns a faster version of the rig rather than hand-drawn animation.

### Team components (co-developed / led by Manish-korde)
DCGAN, Autoencoder, VAE similarity search, FastAPI + React app and Docker deployment.

---

## 🧠 Master Pipeline Architecture

```text
                  GAMEFORGE AI PIPELINE
                           │
                  Natural-Language Request
                           │
                           ▼
                      TRANSFORMER
                    Semantic Planning
                           │
                           ▼
            Structured Game Specification (JSON)
                           │
           ┌───────────────┼────────────────┐
           │               │                │
           ▼               ▼                ▼
       DCGAN         Visual Generation   Asset Library
    Novel Sprite
    Generation
                           │
                     ┌─────┴─────┐
                     ▼           ▼
                    AE           VAE
                     │           │
               Reconstruction   Latent
               / QA Screening  Representation
                     │           │
                     └─────┬─────┘
                           ▼
                     Asset Analysis
```

---

## 🔬 Core Model Components & Technical Applications

Our project presents **four distinct model components**, each fulfilling a technically defensible, non-overlapping role:

### 1. Transformer Model (Semantic Planning)
* **Role**: Translates free-form natural language game requests into structured game specification sheets (identifying themes, character types, environment features, weapon properties, and NPC attributes).
* **Technical Value**: Provides the natural language interface and semantic structuring layer that grounds downstream asset requirements.

### 2. DCGAN (Unconditional Novel Sprite Generator) ✅ Trained
* **Dataset**: Trained on **282,511 2D Game Asset Sprites (Alucard)**, resolution **64×64 RGBA**.
* **Architecture**: Deep Convolutional GAN — `z ∈ ℝ^100 → 4× Conv2DTranspose → 64×64×4 Tanh`.
* **Training**: 50 epochs, 99,300 global optimizer steps, ~5.67 hours on Kaggle P100.
* **Why Unconditional**: Alucard category labels are noisy/unvalidated. Unconditional DCGAN learns the full sprite visual distribution from pixels alone, avoiding cGAN class collapse.
* **Defensible Application**: **Novel 2D Sprite Synthesis** — brand-new RGBA game sprites generated from pure noise `z ~ N(0,I)`, genuinely unseen in training. Distinct from AE (reconstruction) and VAE (latent interpolation).
* **Artifacts**: `models/DCGAN_280K_COMPLETE/weights/generator_latest.keras`

### 3. Autoencoder (AE)
* **Dataset**: Trained & evaluated on the **282,511 2D Game Asset Dataset (Alucard)** ($128 \times 128 \times 4$ RGBA).
* **Role**: Deterministic baseline compression and high-precision pixel-art reconstruction.
* **Performance**: Median MSE `0.000782`, PSNR `28.52 dB`, SSIM `0.9248`.
* **Defensible Application**: **Data Quality Verification & Outlier Screening**. Automatically surfaces the highest-error 5% of assets exceeding the empirical $95\text{th}$ percentile MSE threshold (`0.001313`) for review.

### 4. Variational Autoencoder (VAE)
* **Dataset**: Trained & evaluated on the **282,511 2D Game Asset Dataset (Alucard)**.
* **Active Search Index**: **25,000 Pre-Indexed Latent Vectors** (`latent_gallery_index_25k.npy`) serving distance matching in **< 1ms**.
* **Role**: Probabilistic latent representation learning ($z \sim \mathcal{N}(\mu, \sigma^2)$).
* **Performance**: Evaluated on a 5,000-sprite candidate pool, achieving **84.40% KNN category accuracy** (a **+19.21 percentage point lift** over the 65.19% majority-class baseline) and **62.80% Precision@5** retrieval precision.
* **Defensible Application**: **Probabilistic Latent Space Analysis, Visual Similarity Search, & Controlled Real-Sprite Morphing (A → B)**.

---

## 📊 The Single Dataset Story: 282K Game Assets

To ensure a controlled, fair scientific comparison, **all visual models (AE, VAE, DCGAN) were trained on the exact same 282,511 dataset**:

```text
               282,511 Game Assets (Alucard Dataset)
                         │
          ┌──────────────┼──────────────┐
          ↓              ↓              ↓
         AE             VAE          DCGAN
          ↓              ↓              ↓
   Reconstruction   Continuous    Novel Sprite
   MSE / QA (P95)  Latent Space  Synthesis
(Median 0.000782) (84.4% KNN)  (50ep / 99,300 steps)
```

### Dataset Specifications:
- **Source**: 282,511 2D Pixel-Art Game Sprites (Alucard Dataset `evilsocket/alucard-sprites`).
- **Dimensions**: $128 \times 128 \times 4$ (RGBA with transparent alpha mask) for AE/VAE; $64 \times 64 \times 4$ for DCGAN.
- **100.00% Category Breakdown**: Characters (**65.19%**), Unknown/Misc (**15.84%**), Items (**6.39%**), Enemies (**5.75%**), Weapons (**4.38%**), Tiles (**1.92%**), Props (**0.35%**), Effects (**0.18%**).
- **Project Status & Ground Truth**: See [CURRENT_PROJECT_STATUS.md](file:///c:/Users/manis/OneDrive/Desktop/Prompt_to_game_asset_generator/documentation/CURRENT_PROJECT_STATUS.md) for the definitive status document.

---

## 🚀 How to Run the Project

The project consists of a Python FastAPI backend and a React/Vite web GUI.

### 1. AI Backend (FastAPI)
```bash
cd backend
# Activate virtual environment
.\venv\Scripts\activate
# Start Uvicorn server
uvicorn app:app --port 8000
```
- Status Endpoint: `http://localhost:8000/status`
- Similarity Search Endpoint: `http://localhost:8000/find_similar_vae`

### 2. React Web GUI
```bash
cd gui
npm install
npm run dev
```
Open `http://localhost:5173` in your browser.

### 3. DCGAN Inference (Standalone)
```python
import tensorflow as tf
import numpy as np

generator = tf.keras.models.load_model(
    'models/DCGAN_280K_COMPLETE/weights/generator_latest.keras', compile=False
)
z = np.random.normal(0, 1, size=(16, 100)).astype(np.float32)
sprites = np.clip((generator.predict(z) + 1.0) / 2.0, 0.0, 1.0)  # 16 × 64×64×4 RGBA
```

---

## ⚖️ Ethical Deployment & Responsible AI

1. **Dataset Licensing & Credit**: Attribution to original 2D pixel-art creators (`evilsocket/alucard-sprites`, CC-BY-NC-SA 4.0).
2. **Human-in-the-Loop Oversight**: AI serves as an assistive workflow tool for indie game developers rather than replacing human artists.
3. **Responsible Content Boundaries**: Built-in prompt filtering to enforce safe, non-infringing asset generation.
