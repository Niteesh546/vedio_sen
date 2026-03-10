---
title: Video Sentiment Analyzer
emoji: 🎬
colorFrom: yellow
colorTo: red
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# 🎬 Video Sentiment & Emotion Analyzer

A **multimodal deep learning** system that analyzes videos to detect emotions and sentiment by combining three modalities:

| Modality | Model | Output |
|----------|-------|--------|
| **Text** | BERT (base-uncased) + Whisper STT | 128-dim features |
| **Video** | R3D-18 (3D ResNet) | 128-dim features |
| **Audio** | CNN on Mel Spectrogram | 128-dim features |

The features are fused and passed through two classification heads:
- **Emotion** → 7 classes: anger, disgust, sadness, joy, neutral, surprise, fear
- **Sentiment** → 3 classes: negative, neutral, positive

## 🚀 Quick Start

### Run Locally

```bash
# Clone the repo
git clone https://github.com/Niteesh546/vedio_sen.git
cd vedio_sen

# Install dependencies
pip install -r requirements.txt

# Launch the web app
python app.py
```

Then open http://localhost:7860 in your browser.

### Run with Docker

```bash
docker build -t vedio-sen .
docker run -p 7860:7860 vedio-sen
```

### CLI Inference

```bash
python inference.py path/to/your/video.mp4
```

## 🏗️ Architecture

```
Video Input (.mp4)
    ├── Whisper STT ──→ Text ──→ BERT Encoder ──→ 128-dim
    ├── Frame Extraction ──→ 30 frames ──→ R3D-18 ──→ 128-dim
    └── Audio Extraction ──→ Mel Spec ──→ CNN Encoder ──→ 128-dim
                                                              │
                                              Concatenation (384-dim)
                                                              │
                                                    Fusion Layer (256)
                                                       ┌──────┴──────┐
                                                  Emotion (7)    Sentiment (3)
```

## 📊 Dataset

Trained on the [MELD dataset](https://affective-meld.github.io/) (Multimodal EmotionLines Dataset) — a multimodal multi-party dataset for emotion recognition in conversations from the TV show *Friends*.

## 📁 Project Structure

```
├── app.py                 # Gradio web UI
├── inference.py           # Video analysis pipeline
├── Dockerfile             # Docker deployment
├── requirements.txt       # Python dependencies
├── model.pth              # Trained model weights (after training)
└── training/
    ├── models.py          # Model architecture
    ├── meld_dataset.py    # MELD dataset loader
    ├── train.py           # Training script
    ├── count_parameters.py
    └── test_logging.py
```

## 📝 License

MIT
