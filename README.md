# Musical Chord Root Recognition with CNN

A deep learning project for recognizing musical chord root notes from audio using Convolutional Neural Networks. This streamlined implementation is optimized for Kaggle training with CSV-based feature storage.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red.svg)
![License](https://img.shields.io/badge/license-Educational-green.svg)

## Overview

This project trains CNN models to predict the root note of musical chords from 1-bar audio clips. Given an audio sample containing a chord, the model classifies it into one of 12 possible root notes (C, Db, D, Eb, E, F, Gb, G, Ab, A, Bb, B).

**Key Features:**
- Multiple audio feature representations (Chroma, CQT, Hybrid)
- Data augmentation via pitch-shifting (4x dataset expansion)
- Kaggle-ready CSV export for cloud training
- Streamlined single-notebook preprocessing pipeline
- Grouped train/validation splitting to prevent data leakage

**Performance:** 35-55% validation accuracy (baseline: 8.3% random chance)

## Table of Contents

- [How It Works](#how-it-works)
  - [Audio Feature Extraction](#audio-feature-extraction)
  - [Model Architecture](#model-architecture)
  - [Training Strategy](#training-strategy)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Quick Start](#quick-start)
  - [Detailed Workflow](#detailed-workflow)
- [Feature Types](#feature-types)
- [Data Augmentation](#data-augmentation)
- [Model Training](#model-training)
- [Requirements](#requirements)
- [Documentation](#documentation)

---

## How It Works

### Audio Feature Extraction

The system converts raw audio into spectral representations that capture harmonic and pitch information. Four different feature types are supported, each offering different trade-offs between detail and computational efficiency.

#### 1. Chromagram (12 bins)
<img src="feature_visualizations/chroma_example.png" alt="Chromagram visualization" width="600"/>

**What it is:** Octave-invariant pitch class representation, collapsing all frequencies into 12 bins (one per musical note).

**Why it works:** Chords with the same root note share similar chroma patterns regardless of voicing or octave.

**Best for:** Fast training, root note detection

#### 2. Hybrid Features (36 bins)
<img src="feature_visualizations/hybrid_example.png" alt="Hybrid features visualization" width="600"/>

**What it is:** Combination of low-frequency CQT (24 bins covering 2 octaves) stacked with chromagram (12 bins).

**Why it works:** Preserves critical bass note information (where the root typically resides) while maintaining harmonic context.

**Best for:** Overall best performance, balanced approach

#### 3. CQT-60 (60 bins)
<img src="feature_visualizations/cqt60_example.png" alt="CQT-60 visualization" width="600"/>

**What it is:** Constant-Q Transform covering 5 octaves with logarithmic frequency spacing.

**Why it works:** Captures harmonic relationships with musically-meaningful frequency resolution.

**Best for:** Detailed harmonic analysis without excessive computation

#### 4. CQT-84 (84 bins)
<img src="feature_visualizations/cqt84_example.png" alt="CQT-84 visualization" width="600"/>

**What it is:** Full-range CQT spanning 7 octaves (C1 to B7).

**Why it works:** Maximum frequency detail captures all harmonic overtones and upper extensions.

**Best for:** Maximum information preservation (slower training)

---

### Model Architecture

The CNN architecture is specifically designed for audio chord recognition:

```
Input: (batch, 1, n_bins, 87 frames)
    ↓
Conv2D (5×1 kernel) + BatchNorm + ReLU  ← Vertical kernel captures harmonics
    ↓
MaxPool2D (2×2)
    ↓
Conv2D (3×3) + BatchNorm + ReLU
    ↓
MaxPool2D (2×2)
    ↓
Conv2D (3×3) + BatchNorm + ReLU
    ↓
MaxPool2D (2×2)
    ↓
Conv2D (3×3) + BatchNorm + ReLU
    ↓
AdaptiveAvgPool2D (1×1)  ← Handles variable input sizes
    ↓
Flatten → FC(256) → Dropout → FC(128) → Dropout → FC(12)
    ↓
Output: 12 class logits (one per root note)
```

**Key Design Choices:**
- **5×1 first kernel:** Captures vertical harmonic relationships in the spectrogram
- **Asymmetric pooling:** (2×2) preserves temporal resolution early in the network
- **Adaptive pooling:** Same architecture works with 12, 36, 60, or 84 input bins
- **Batch normalization:** Stabilizes training and accelerates convergence
- **Dropout:** Prevents overfitting on limited musical data

**Model size:** ~107K-250K parameters depending on input feature type

---

### Training Strategy

#### Data Augmentation: Pitch Shifting

Each audio sample is pitch-shifted by **-2, 0, +2, +4 semitones**, increasing the dataset size by 4x while preserving musical relationships.

```
Original: beatrice_bar01.wav → Label: F:maj
    ↓ shift -2 semitones
Augmented: beatrice_bar01_shift-2.wav → Label: Eb:maj
    ↓ shift +2 semitones
Augmented: beatrice_bar01_shift+2.wav → Label: G:maj
    ↓ shift +4 semitones
Augmented: beatrice_bar01_shift+4.wav → Label: A:maj
```

**Why this works:** Pitch-shifting preserves harmonic structure while creating "new" training examples with different root notes.

#### Grouped Train/Validation Split

**Critical:** Standard random splitting would cause data leakage!

Since `beatrice_bar01.wav` and `beatrice_bar01_shift+2.wav` are the same audio (just transposed), they must stay together in either the training or validation set.

```python
# Group all augmentations of the same bar together
groups = df['id'].str.replace(r'_shift.*', '', regex=True)
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(splitter.split(X, y, groups))
```

This ensures the model is evaluated on truly unseen musical content, not just transposed versions of training data.

---

## Project Structure

```
Chord_Recognition/
├── README.md                      # This file
├── CLAUDE.md                      # Complete technical documentation
├── WORKFLOW.md                    # Detailed pipeline visualization
├── requirements.txt               # Python dependencies
│
├── data_preprocessing.ipynb       # Single notebook for all preprocessing
├── model_training.ipynb           # CNN training notebook (optional local training)
│
├── data_input/
│   ├── 00_raw/                    # Input: Ableton exports
│   │   └── song_name/
│   │       ├── song_name_chords   # Chord progression text file
│   │       └── song_name_clips/   # Bar-by-bar audio slices (.wav)
│   └── 00_input/                  # Input: MusicXML + full audio (future)
│
├── data_output/                   # CSV files ready for Kaggle upload
│   ├── chroma_features.csv        # 12 × 87 = 1,044 features per sample
│   ├── hybrid_features.csv        # 36 × 87 = 3,132 features per sample
│   ├── cqt60_features.csv         # 60 × 87 = 5,220 features per sample
│   └── cqt84_features.csv         # 84 × 87 = 7,308 features per sample
│
└── feature_visualizations/        # Example feature plots
    ├── chroma_example.png
    ├── hybrid_example.png
    ├── cqt60_example.png
    └── cqt84_example.png
```

---

## Installation

### Prerequisites
- Python 3.8 or higher
- ~2GB free disk space for dependencies
- (Optional) CUDA-capable GPU for faster training

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/chord-recognition.git
cd chord-recognition

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

Core libraries:
- **PyTorch** 2.1.0 - Deep learning framework
- **librosa** 0.10.1 - Audio feature extraction
- **soundfile** 0.12.1 - Audio I/O
- **numpy** 1.24.3 - Numerical computing
- **pandas** - CSV data handling
- **tqdm** - Progress bars
- **jupyter** - Notebook interface

See `requirements.txt` for complete list with exact versions.

---

## Usage

### Quick Start

**1. Prepare your data**

Place your audio data in `data_input/00_raw/`:

```
data_input/00_raw/my_song/
├── my_song_chords        # Text file with one chord per line
└── my_song_clips/        # Folder with WAV files
    ├── Slice 1 [...].wav
    ├── Slice 2 [...].wav
    └── ...
```

**Chord file format** (`my_song_chords`):
```
Fmaj7
Gbmaj7
Dm7 | Cm7    # Two chords in one bar (will be split)
Bbm7
Am7b5
```

**2. Run preprocessing**

```bash
jupyter notebook data_preprocessing.ipynb
# Execute all cells
```

This will:
- Load and deduplicate audio slices
- Split bars containing two chords
- Apply pitch-shift augmentation
- Extract all four feature types
- Export CSV files to `data_output/`

**3. Upload to Kaggle**

1. Go to [kaggle.com/datasets](https://www.kaggle.com/datasets)
2. Click "New Dataset"
3. Upload CSV files from `data_output/`
4. Create a new notebook and start training!

---

### Detailed Workflow

#### Step 1: Data Preprocessing (Local)

The `data_preprocessing.ipynb` notebook handles all preprocessing in memory:

```python
# The notebook performs these steps automatically:
# 1. Load raw audio + chord labels
# 2. Deduplicate Ableton exports (removes duplicate slices)
# 3. Split bars with two chords into separate samples
# 4. Apply pitch-shift augmentation (-2, 0, +2, +4 semitones)
# 5. Extract features (Chroma, Hybrid, CQT60, CQT84)
# 6. Flatten features and export to CSV
```

**Output:** Four CSV files in `data_output/`, each containing:
- `id` - Sample identifier (e.g., "beatrice_bar01_shift+2")
- `label` - Full chord label (e.g., "F:maj")
- `root` - Root note only (e.g., "F")
- `quality` - Chord quality (e.g., "maj", "min", "dom")
- `song` - Song name
- `shift` - Pitch shift amount (-2, 0, 2, 4)
- `feature_0` to `feature_N` - Flattened feature array

#### Step 2: Training (Kaggle)

Create a Kaggle notebook and use this template:

```python
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit

# Load data
df = pd.read_csv('../input/your-dataset/chroma_features.csv')

# Extract features and labels
X = df.filter(regex='feature_').values
y = df['root'].map({root: i for i, root in enumerate(
    ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
)}).values

# Grouped split (critical to prevent data leakage!)
groups = df['id'].str.replace(r'_shift.*', '', regex=True)
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(splitter.split(X, y, groups))

# Split data
X_train, X_val = X[train_idx], X[val_idx]
y_train, y_val = y[train_idx], y[val_idx]

# Reshape for CNN: (samples, channels=1, n_bins, n_frames=87)
n_bins = 12  # For chroma; use 36 for hybrid, 60 for CQT60, 84 for CQT84
X_train = X_train.reshape(-1, 1, n_bins, 87)
X_val = X_val.reshape(-1, 1, n_bins, 87)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)
X_val = torch.FloatTensor(X_val)
y_val = torch.LongTensor(y_val)

# Create data loaders
from torch.utils.data import TensorDataset, DataLoader

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define model (see CLAUDE.md for complete ChordCNN implementation)
# model = ChordCNN(num_classes=12, input_bins=n_bins)

# Train your model...
```

---

## Feature Types

| Feature Type | Dimensions | CSV Size* | Training Speed | Typical Accuracy | Use Case |
|--------------|-----------|-----------|----------------|------------------|----------|
| **Chroma** | 12 × 87 | ~50 MB | Fastest | 40-45% | Quick prototyping, root detection |
| **Hybrid** | 36 × 87 | ~150 MB | Fast | **45-55%** | Best overall performance |
| **CQT-60** | 60 × 87 | ~250 MB | Medium | 40-50% | Balanced detail vs. speed |
| **CQT-84** | 84 × 87 | ~350 MB | Slower | 40-50% | Maximum frequency detail |

*Size estimates based on ~1000 augmented samples

**Recommendation:** Start with **Chroma** for fast iteration, then try **Hybrid** for best results.

---

## Data Augmentation

Pitch-shift augmentation is applied with default shifts of **-2, 0, +2, +4 semitones**.

**Benefits:**
- Increases dataset size by 4x
- Provides more varied training examples
- Balances root note distribution across all 12 classes

**Implementation:**
```python
# Handled automatically in the preprocessing notebook
shifts = [-2, 0, 2, 4]  # Configurable

for shift in shifts:
    # Apply pitch shift to audio
    y_shifted = librosa.effects.pitch_shift(y, sr=22050, n_steps=shift)

    # Transpose the chord label
    # Example: F:maj shifted +2 → G:maj
    new_label = transpose_chord(original_label, shift)
```

You can modify the `shifts` list in the notebook to experiment with different augmentation strategies.

---

## Model Training

### Training Configuration

Typical hyperparameters:
- **Optimizer:** Adam (lr=0.001)
- **Loss:** CrossEntropyLoss
- **Batch size:** 32
- **Epochs:** 100 (with early stopping)
- **Learning rate schedule:** ReduceLROnPlateau (factor=0.5, patience=5)
- **Early stopping patience:** 10 epochs

### Expected Performance

| Feature Type | Val Accuracy | Training Time* |
|--------------|--------------|---------------|
| Chroma | 40-45% | ~15 min |
| Hybrid | **45-55%** | ~25 min |
| CQT-60 | 40-50% | ~35 min |
| CQT-84 | 40-50% | ~45 min |

*Using Kaggle GPU (Tesla P100)

**Baseline:** Random guessing = 8.3% (1/12 classes)

### Tips for Better Performance

1. **Start with chroma** - Fast feedback loop for architecture experiments
2. **Use grouped splitting** - Critical to prevent data leakage
3. **Monitor overfitting** - Training accuracy much higher than validation? Increase dropout
4. **Try hybrid features** - Usually best balance of performance and speed
5. **Ensemble methods** - Combine predictions from multiple feature types

---

## Requirements

### System Requirements
- **OS:** Linux, macOS, or Windows
- **RAM:** 8GB minimum (16GB recommended)
- **Storage:** 5GB free space (including dependencies and dataset)
- **GPU:** Optional but recommended for training (Kaggle provides free GPU)

### Input Data Format

**Ableton Export Format** (supported now):
```
song_name/
├── song_name_chords        # Plain text, one chord per line
└── song_name_clips/        # WAV files (any sample rate, mono/stereo)
    ├── Slice 1 [timestamp].wav
    ├── Slice 2 [timestamp].wav
    └── ...
```

**Supported chord symbols:**
- Major: `C`, `Cmaj7`, `Cmaj9`, `C6`, `C69`
- Minor: `Cm`, `Cm7`, `Cm9`, `Cmin7`
- Dominant: `C7`, `C9`, `C13`
- Half-diminished: `Cm7b5`, `Cø7`
- Diminished: `Cdim`, `Cdim7`
- Any root note: C, C#/Db, D, D#/Eb, E, F, F#/Gb, G, G#/Ab, A, A#/Bb, B

**Multi-chord bars:**
```
Dm7 | G7    # Will be split into two samples: Dm7 and G7
```

---

## Documentation

- **`README.md`** (this file) - Overview, usage, and getting started
- **`CLAUDE.md`** - Complete technical documentation with code examples
  - Feature extraction algorithms
  - CNN architecture details
  - Training procedures
  - Chord parsing logic
- **`WORKFLOW.md`** - Visual pipeline diagrams and workflow details

---

## Advantages Over Original Project

| Original Project | Refactored Version |
|------------------|-------------------|
| 5 data directories | 2 directories (input/output) |
| 7 Python scripts | 1 preprocessing notebook |
| .npy files (not portable) | CSV files (Kaggle-ready) |
| Local training only | Cloud training on Kaggle |
| Complex multi-stage pipeline | Streamlined single-notebook workflow |

---

## Future Improvements

Potential enhancements:
- [ ] Full chord quality prediction (maj/min/dom/hdim/dim) - 60 classes
- [ ] Real-time audio chord recognition
- [ ] Transformer-based architecture
- [ ] MusicXML + full audio input support
- [ ] Web-based inference demo
- [ ] Ensemble predictions across feature types

---

## License

Educational project for deep learning coursework.

---

## Acknowledgments

- **librosa** - Comprehensive audio analysis library
- **PyTorch** - Flexible deep learning framework
- **Kaggle** - Free GPU resources for training

---

## Contact

For questions or issues, please open an issue on GitHub.

---

**Happy chord recognition!**
