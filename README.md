# ⚾ Hall of Fame Face Predictor App

An interactive Streamlit app that predicts whether a baseball player "looks like" a Hall of Famer based on their Baseball Reference photo. A fun exploration of image classification with neural networks!

> **⚠️ Disclaimer**: This is a satirical/educational project! A player's appearance has absolutely nothing to do with their baseball ability or Hall of Fame worthiness. This model finds spurious correlations in image data and is meant for entertainment and learning purposes only.

---

## 🎯 Features

- **Player Lookup**: Enter any Baseball Reference player ID to fetch their headshot
- **Live Image Scraping**: Automatically retrieves player photos from Baseball-Reference.com
- **Image Pipeline Visualization**: See each preprocessing step (original → grayscale → 32×32)
- **Neural Network Prediction**: Real-time inference using a trained Keras model
- **Probability Display**: Visual progress bars and confidence metrics
- **Cloudflare Bypass**: Uses `curl_cffi` for reliable image fetching

---

## 📁 Project Structure

```
hof-predictor-app/
├── app.py                  # Main Streamlit application
├── pyproject.toml          # Project dependencies (uv/pip)
├── uv.lock                 # Locked dependencies
├── README.md               # This file
│
├── model/
│   ├── train_model.py      # Model training script
│   └── hof_model.keras     # Pre-trained model weights
│
├── utils/
│   ├── __init__.py
│   ├── scraper.py          # Baseball Reference image scraper
│   └── preprocessor.py     # Image preprocessing utilities
│
└── data/
    └── Hall_of_Fame_Eligible/  # Cached player images
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager (recommended)

### Installation

1. **Clone and navigate to the project:**
   ```bash
   cd hof-predictor-app
   ```

2. **Install dependencies with uv:**
   ```bash
   uv sync
   ```

3. **Install curl_cffi for reliable scraping (bypasses Cloudflare):**
   ```bash
   uv pip install curl_cffi
   ```

4. **Run the app:**
   ```bash
   uv run streamlit run app.py
   ```

5. **Open in browser:**
   - Local: http://localhost:8501
   - Network: http://your-ip:8501

---

## 🎮 Usage Guide

### Finding Player IDs

Player IDs come from Baseball Reference URLs. For example:
- `https://www.baseball-reference.com/players/a/aaronha01.shtml` → ID: **aaronha01**
- `https://www.baseball-reference.com/players/b/bondsba01.shtml` → ID: **bondsba01**

### Example Player IDs to Try

| Player ID | Player Name | Status |
|-----------|-------------|--------|
| `aaronha01` | Hank Aaron | Hall of Famer ✅ |
| `bondsba01` | Barry Bonds | Not in HOF ❌ |
| `rosepe01` | Pete Rose | Banned 🚫 |
| `jeterde01` | Derek Jeter | Hall of Famer ✅ |
| `griffke02` | Ken Griffey Jr. | Hall of Famer ✅ |
| `pujolal01` | Albert Pujols | Not yet eligible ⏳ |
| `troutmi01` | Mike Trout | Active player 🔄 |
| `clemero02` | Roger Clemens | Not in HOF ❌ |

---

## 🧠 Model Architecture

The neural network is a simple fully-connected classifier:

```
Input Layer:     1024 neurons (32×32 flattened grayscale image)
    ↓
Dense Layer 1:   256 neurons (ReLU activation)
Dropout:         30%
    ↓
Dense Layer 2:   128 neurons (ReLU activation)
Dropout:         30%
    ↓
Output Layer:    2 neurons (Softmax: [Not HOF, HOF])
```

### Training Details

- **Optimizer**: RMSprop
- **Loss**: Binary Crossentropy
- **Training Data**: ~100+ player images from Hall of Fame eligible players
- **Input**: 32×32 grayscale images normalized to [0, 1]

---

## 🔧 Technical Implementation

### Image Scraping (`utils/scraper.py`)

The scraper handles Baseball Reference's Cloudflare protection:

1. **curl_cffi** (preferred): Uses Chrome TLS fingerprinting to bypass bot detection
2. **Fallback**: Standard `requests` library with retry logic

Key functions:
- `scrape_player_image(player_id)` → Returns PIL Image
- `get_player_name(player_id)` → Returns player's display name
- `download_player_image(player_id)` → Saves image locally

### Image Preprocessing (`utils/preprocessor.py`)

```python
def preprocess_image(img):
    # 1. Convert to grayscale
    img_gray = img.convert('L')
    
    # 2. Resize to 32×32
    img_resized = img_gray.resize((32, 32))
    
    # 3. Normalize to [0, 1]
    img_array = np.array(img_resized) / 255.0
    
    # 4. Flatten for model input
    img_flat = img_array.flatten()  # Shape: (1024,)
    
    return img_flat, img_gray, img_resized
```

### Placeholder Detection

The scraper includes smart caching with placeholder detection:
- Checks for cached images before fetching
- Detects placeholder images (< 5 unique colors in first 100 pixels)
- Automatically deletes placeholders and re-fetches

---

## 🚀 Deployment

### Posit Connect Cloud

1. **Generate requirements.txt:**
   ```bash
   uv pip compile pyproject.toml -o requirements.txt
   ```

2. **Add curl_cffi to requirements.txt** (if not included):
   ```
   curl_cffi>=0.5.0
   ```

3. **Deploy via Posit Connect Cloud UI or rsconnect CLI**

### Docker (Alternative)

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY . .

RUN pip install uv && uv sync
RUN uv pip install curl_cffi

EXPOSE 8501
CMD ["uv", "run", "streamlit", "run", "app.py", "--server.address=0.0.0.0"]
```

---

## 🔄 Retraining the Model

If you want to retrain the model with different data:

1. **Prepare training images:**
   - Place HOF player images in a folder with prefix `hof_`
   - Place non-HOF player images with prefix `nothof_`
   - Update the `data_dir` path in `model/train_model.py`

2. **Run training:**
   ```bash
   uv run python model/train_model.py
   ```

3. **Model will be saved to:** `model/hof_model.keras`

---

## 🐛 Troubleshooting

### "Couldn't find a tree builder with the features you requested: lxml"

Install lxml:
```bash
uv pip install lxml
```

### Images showing as gray placeholders

1. Install curl_cffi for Cloudflare bypass:
   ```bash
   uv pip install curl_cffi
   ```

2. Delete cached placeholder images:
   ```bash
   rm data/Hall_of_Fame_Eligible/eligibility_*.jpg
   ```

3. Restart the app

### "Model not found" error

Run the training script first:
```bash
uv run python model/train_model.py
```

---

## 📊 Implementation Plan / Roadmap

### ✅ Phase 1: Core Functionality (Complete)
- [x] Basic Streamlit app structure
- [x] Baseball Reference image scraping
- [x] Image preprocessing pipeline
- [x] Neural network model training
- [x] Prediction display with probabilities

### ✅ Phase 2: Reliability Improvements (Complete)
- [x] Cloudflare bypass with curl_cffi
- [x] Placeholder image detection
- [x] Automatic cache invalidation
- [x] Fallback parsing (lxml → html.parser)

### 🔄 Phase 3: Future Enhancements (Planned)
- [ ] Batch prediction for multiple players
- [ ] Side-by-side player comparison
- [ ] Historical player lookup (by name search)
- [ ] Model explainability (feature importance visualization)
- [ ] Career statistics integration
- [ ] Confidence calibration improvements
- [ ] Alternative model architectures (CNN, transfer learning)

### 🎯 Phase 4: Deployment & Scaling
- [ ] Posit Connect Cloud deployment
- [ ] Docker containerization
- [ ] CI/CD pipeline
- [ ] Rate limiting for scraper
- [ ] Image caching layer (Redis/S3)

---

## 📝 Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web application framework |
| `tensorflow` | Neural network training & inference |
| `keras` | High-level neural network API |
| `pillow` | Image processing |
| `numpy` | Numerical operations |
| `beautifulsoup4` | HTML parsing |
| `lxml` | Fast XML/HTML parser |
| `curl_cffi` | Cloudflare bypass (TLS fingerprinting) |
| `requests` | HTTP client (fallback) |

---

## 📜 License

This project is for educational and entertainment purposes only.

---

## 🙏 Acknowledgments

- [Baseball Reference](https://www.baseball-reference.com) for player data and images
- The Streamlit team for an excellent framework
- TensorFlow/Keras for accessible deep learning

---

<div align="center">

**Made with ❤️ and a healthy dose of skepticism**

*Remember: Judging Hall of Fame worthiness by facial features is absurd. That's the point!*

</div>
